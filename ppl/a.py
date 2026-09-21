"""
debug.py - Fixed Debug Script with consistent method naming
"""

import torch
import torch.nn.functional as F
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


def extract_valid_tokens_from_left_padded(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Extract valid tokens from left-padded input."""
    batch_size = input_ids.shape[0]
    device = input_ids.device

    valid_lengths = attention_mask.sum(dim=1).to(torch.int32)
    max_valid_len = valid_lengths.max().item()

    valid_input_ids = torch.zeros(batch_size, max_valid_len, dtype=input_ids.dtype, device=device)

    for b in range(batch_size):
        vlen = valid_lengths[b].item()
        if vlen > 0:
            valid_input_ids[b, :vlen] = input_ids[b, -vlen:]

    return valid_input_ids, valid_lengths


def generate_diffusion_quality(
    self,
    input_ids: torch.LongTensor,
    mask_token_id: int = None,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 50,
    steps: int = 10,
    temperature: float = 1.0,
    block_size: int = 8,
    temperature_schedule: str = "cosine",
    final_temperature: float = 0.1,
    use_confidence_masking: bool = False,
    confidence_threshold: float = 0.9,
    min_refinement_steps: int = 2,
    use_gumbel: bool = True,
    gumbel_temperature: float = 0.7,
    **kwargs
) -> torch.LongTensor:
    """Quality-optimized generation with proper left-padding support."""
    mask_token_id = mask_token_id or self.config.mask_token_id
    batch_size, original_padded_len = input_ids.shape
    device, dtype = input_ids.device, self.lm_head.weight.dtype

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    # Extract valid tokens from left-padded input
    valid_input_ids, valid_lengths = extract_valid_tokens_from_left_padded(input_ids, attention_mask)
    max_valid_len = valid_lengths.max().item()

    def get_scheduled_temperature(step: int, total_steps: int, base_temp: float, final_temp: float) -> float:
        if temperature_schedule == "constant" or total_steps <= 1:
            return base_temp
        progress = step / max(total_steps - 1, 1)
        if temperature_schedule == "linear":
            return base_temp + (final_temp - base_temp) * progress
        elif temperature_schedule == "cosine":
            return final_temp + (base_temp - final_temp) * (1 + math.cos(math.pi * progress)) / 2
        return base_temp

    # Initialize KV cache (cache_leftpad=0 for all, valid tokens at buffer start)
    cache = self._init_kv_cache(batch_size, max_valid_len + max_new_tokens, device, dtype)

    # Prefill with valid tokens
    self.forward(
        input_ids=valid_input_ids,
        attention_mask=None,
        past_key_values=cache.get_all_layer_caches(),
        cache_seqlens=cache.get_cache_seqlens(),
        cache_leftpad=cache.get_cache_leftpad(),
        use_cache=True,
        causal=True,
        return_dict=True
    )
    cache.set_cache_seqlens(valid_lengths)

    committed = valid_input_ids.clone()
    committed_lens = valid_lengths.clone()
    generated = []
    num_generated = 0

    while num_generated < max_new_tokens:
        curr_block_size = min(block_size, max_new_tokens - num_generated)

        block = torch.full(
            (batch_size, curr_block_size),
            mask_token_id,
            dtype=input_ids.dtype,
            device=device
        )

        block_confidence = torch.zeros(batch_size, curr_block_size, device=device)
        converged = False
        last_block_state = None
        effective_steps = min(steps, curr_block_size + min_refinement_steps)

        for step in range(effective_steps):
            ar_pointer = step

            if ar_pointer >= curr_block_size and step >= min_refinement_steps:
                converged = (block == last_block_state).all() if last_block_state is not None else False
                break

            current_gumbel_temp = get_scheduled_temperature(
                step, effective_steps, gumbel_temperature, final_temperature
            )

            cache.set_cache_seqlens(committed_lens - 1)

            last_tokens = torch.stack([
                committed[b, committed_lens[b] - 1] 
                for b in range(batch_size)
            ]).unsqueeze(1)
            block_input = torch.cat([last_tokens, block], dim=1)

            outputs = self.forward(
                input_ids=block_input,
                past_key_values=cache.get_all_layer_caches(),
                cache_seqlens=cache.get_cache_seqlens(),
                cache_leftpad=cache.get_cache_leftpad(),
                use_cache=True,
                causal=True,
                return_dict=True
            )

            relevant_logits = outputs.logits[:, 1:curr_block_size + 1, :]

            if use_gumbel and current_gumbel_temp > 0:
                gumbel_logits = self._add_gumbel_noise(relevant_logits, current_gumbel_temp)
                candidate_tokens = torch.argmax(gumbel_logits, dim=-1).to(input_ids.dtype)
            elif temperature > 0:
                gumbel_logits = self._add_gumbel_noise(relevant_logits, temperature)
                candidate_tokens = torch.argmax(gumbel_logits, dim=-1).to(input_ids.dtype)
            else:
                candidate_tokens = torch.argmax(relevant_logits, dim=-1).to(input_ids.dtype)

            if use_confidence_masking:
                probs = F.softmax(relevant_logits.float(), dim=-1)
                new_confidence, _ = probs.max(dim=-1)
            else:
                new_confidence = torch.ones(batch_size, curr_block_size, device=device)

            old_block = block.clone()

            if use_confidence_masking and step > 0:
                update_mask = block_confidence < confidence_threshold
                update_mask[:, min(ar_pointer, curr_block_size - 1):] = True
            else:
                update_mask = torch.zeros(batch_size, curr_block_size, dtype=torch.bool, device=device)
                if ar_pointer < curr_block_size:
                    update_mask[:, ar_pointer:] = True
                else:
                    update_mask[:, -1:] = True

            block = torch.where(update_mask, candidate_tokens, old_block)
            block_confidence = torch.where(update_mask, new_confidence, block_confidence)
            last_block_state = old_block

            no_changes = (block == old_block).all()
            all_confident = (block_confidence >= confidence_threshold).all() if use_confidence_masking else False

            if step >= min_refinement_steps - 1:
                if no_changes:
                    converged = True
                    break
                if all_confident and use_confidence_masking:
                    converged = True
                    break

        # Commit block
        if not converged:
            cache.set_cache_seqlens(committed_lens - 1)
            last_tokens = torch.stack([
                committed[b, committed_lens[b] - 1] 
                for b in range(batch_size)
            ]).unsqueeze(1)

            self.forward(
                input_ids=torch.cat([last_tokens, block], dim=1),
                past_key_values=cache.get_all_layer_caches(),
                cache_seqlens=cache.get_cache_seqlens(),
                cache_leftpad=cache.get_cache_leftpad(),
                use_cache=True,
                causal=True,
                return_dict=True
            )

        committed_lens = committed_lens + curr_block_size
        cache.set_cache_seqlens(committed_lens)
        committed = torch.cat([committed, block], dim=1)
        generated.append(block)
        num_generated += curr_block_size

    return torch.cat([input_ids] + generated, dim=1)


def generate_diffusion_block_kvcache_corrected(
    self,
    input_ids: torch.LongTensor,
    mask_token_id: int = None,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 50,
    steps: int = 10,
    temperature: float = 1.0,
    block_size: int = 8,
    use_gumbel: bool = True,
    gumbel_temperature: float = 0.7,
    **kwargs
) -> torch.LongTensor:
    """Original block-based KV cache generation with proper left-padding support."""
    mask_token_id = mask_token_id or self.config.mask_token_id
    batch_size, original_padded_len = input_ids.shape
    device, dtype = input_ids.device, self.lm_head.weight.dtype

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    # Extract valid tokens
    valid_input_ids, valid_lengths = extract_valid_tokens_from_left_padded(input_ids, attention_mask)
    max_valid_len = valid_lengths.max().item()

    # Initialize KV cache
    cache = self._init_kv_cache(batch_size, max_valid_len + max_new_tokens, device, dtype)

    # Prefill
    self.forward(
        input_ids=valid_input_ids,
        attention_mask=None,
        past_key_values=cache.get_all_layer_caches(),
        cache_seqlens=cache.get_cache_seqlens(),
        cache_leftpad=cache.get_cache_leftpad(),
        use_cache=True,
        causal=True,
        return_dict=True
    )
    cache.set_cache_seqlens(valid_lengths)

    committed = valid_input_ids.clone()
    committed_lens = valid_lengths.clone()
    generated = []
    num_generated = 0

    while num_generated < max_new_tokens:
        curr_block_size = min(block_size, max_new_tokens - num_generated)

        block = torch.full(
            (batch_size, curr_block_size),
            mask_token_id,
            dtype=input_ids.dtype,
            device=device
        )

        for step in range(steps):
            cache.set_cache_seqlens(committed_lens - 1)

            last_tokens = torch.stack([
                committed[b, committed_lens[b] - 1] 
                for b in range(batch_size)
            ]).unsqueeze(1)
            block_input = torch.cat([last_tokens, block], dim=1)

            outputs = self.forward(
                input_ids=block_input,
                past_key_values=cache.get_all_layer_caches(),
                cache_seqlens=cache.get_cache_seqlens(),
                cache_leftpad=cache.get_cache_leftpad(),
                use_cache=True,
                causal=True,
                return_dict=True
            )

            relevant_logits = outputs.logits[:, 1:curr_block_size + 1, :]

            if use_gumbel and gumbel_temperature > 0:
                gumbel_logits = self._add_gumbel_noise(relevant_logits, gumbel_temperature)
                block = torch.argmax(gumbel_logits, dim=-1).to(input_ids.dtype)
            elif temperature > 0:
                gumbel_logits = self._add_gumbel_noise(relevant_logits, temperature)
                block = torch.argmax(gumbel_logits, dim=-1).to(input_ids.dtype)
            else:
                block = torch.argmax(relevant_logits, dim=-1).to(input_ids.dtype)

        # Commit
        cache.set_cache_seqlens(committed_lens - 1)
        last_tokens = torch.stack([
            committed[b, committed_lens[b] - 1] 
            for b in range(batch_size)
        ]).unsqueeze(1)

        self.forward(
            input_ids=torch.cat([last_tokens, block], dim=1),
            past_key_values=cache.get_all_layer_caches(),
            cache_seqlens=cache.get_cache_seqlens(),
            cache_leftpad=cache.get_cache_leftpad(),
            use_cache=True,
            causal=True,
            return_dict=True
        )

        committed_lens = committed_lens + curr_block_size
        cache.set_cache_seqlens(committed_lens)
        committed = torch.cat([committed, block], dim=1)
        generated.append(block)
        num_generated += curr_block_size

    return torch.cat([input_ids] + generated, dim=1)


def main():
    DIFFUSION_MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_main_exp/checkpoint-77335"
    HELLASWAG_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/Rowan/hellaswag/main"

    device = "cuda:0"
    MAX_NEW_TOKENS = 32
    NUM_SAMPLES = 5

    CONFIGS = {
        "original": {
            "method": "original",
            "block_size": 16,
            "steps": 16,
            "gumbel_temperature": 0.7,
        },
        "quality_cosine": {
            "method": "quality",
            "block_size": 16,
            "steps": 16,
            "gumbel_temperature": 0.7,
            "temperature_schedule": "cosine",
            "final_temperature": 0.2,
            "use_confidence_masking": False,
        },
    }

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(DIFFUSION_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        DIFFUSION_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device).eval()

    import types
    model.generate_diffusion_quality = types.MethodType(generate_diffusion_quality, model)
    model.generate_diffusion_block_kvcache_corrected = types.MethodType(
        generate_diffusion_block_kvcache_corrected, model
    )

    mask_token_id = model.config.mask_token_id
    print(f"mask_token_id: {mask_token_id}")

    print("Loading dataset...")
    dataset = load_dataset(HELLASWAG_PATH)
    prompts = list(dataset["validation"]["ctx"][:NUM_SAMPLES])

    # ============================================================
    # TEST 1: Single sample inference
    # ============================================================
    print(f"\n{'='*80}")
    print("TEST 1: SINGLE SAMPLE INFERENCE (BASELINE)")
    print(f"{'='*80}")

    for i, prompt in enumerate(prompts[:2]):
        print(f"\n--- Sample {i} ---")
        print(f"Prompt: {prompt[:80]}...")

        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2560,
            add_special_tokens=False
        ).to(device)

        print(f"input_ids shape: {inputs['input_ids'].shape}")
        prompt_len = inputs["input_ids"].shape[1]

        for config_name, config in CONFIGS.items():
            with torch.inference_mode():
                if config["method"] == "original":
                    output_ids = model.generate_diffusion_block_kvcache_corrected(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        mask_token_id=mask_token_id,
                        max_new_tokens=MAX_NEW_TOKENS,
                        block_size=config["block_size"],
                        steps=config["steps"],
                        temperature=0.7,
                        use_gumbel=True,
                        gumbel_temperature=config["gumbel_temperature"],
                    )
                elif config["method"] == "quality":
                    output_ids = model.generate_diffusion_quality(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        mask_token_id=mask_token_id,
                        max_new_tokens=MAX_NEW_TOKENS,
                        block_size=config["block_size"],
                        steps=config["steps"],
                        temperature=0.7,
                        use_gumbel=True,
                        gumbel_temperature=config["gumbel_temperature"],
                        temperature_schedule=config.get("temperature_schedule", "constant"),
                        final_temperature=config.get("final_temperature", 0.2),
                        use_confidence_masking=config.get("use_confidence_masking", False),
                    )

            generated_ids = output_ids[:, prompt_len:]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"  {config_name}: {decoded[:100]}...")

    # ============================================================
    # TEST 2: Batched inference with padding
    # ============================================================
    print(f"\n{'='*80}")
    print("TEST 2: BATCHED INFERENCE WITH LEFT-PADDING")
    print(f"{'='*80}")

    batch_prompts = prompts[:NUM_SAMPLES]

    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2560,
        add_special_tokens=False
    ).to(device)

    print(f"\nBatch size: {len(batch_prompts)}")
    print(f"input_ids shape: {inputs['input_ids'].shape}")
    print(f"attention_mask shape: {inputs['attention_mask'].shape}")

    # Show padding info
    for i in range(len(batch_prompts)):
        num_pad = (inputs['attention_mask'][i] == 0).sum().item()
        num_valid = (inputs['attention_mask'][i] == 1).sum().item()
        print(f"  Sample {i}: leftpad={num_pad}, valid_tokens={num_valid}")

    prompt_len = inputs["input_ids"].shape[1]

    for config_name, config in CONFIGS.items():
        print(f"\n--- {config_name.upper()} (BATCHED) ---")

        with torch.inference_mode():
            if config["method"] == "original":
                output_ids = model.generate_diffusion_block_kvcache_corrected(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    mask_token_id=mask_token_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    block_size=config["block_size"],
                    steps=config["steps"],
                    temperature=0.7,
                    use_gumbel=True,
                    gumbel_temperature=config["gumbel_temperature"],
                )
            elif config["method"] == "quality":
                output_ids = model.generate_diffusion_quality(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    mask_token_id=mask_token_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    block_size=config["block_size"],
                    steps=config["steps"],
                    temperature=0.7,
                    use_gumbel=True,
                    gumbel_temperature=config["gumbel_temperature"],
                    temperature_schedule=config.get("temperature_schedule", "constant"),
                    final_temperature=config.get("final_temperature", 0.2),
                    use_confidence_masking=config.get("use_confidence_masking", False),
                )

        generated_ids = output_ids[:, prompt_len:]
        generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, gen in enumerate(generations):
            print(f"  Sample {i}: {gen[:80]}...")

    # ============================================================
    # TEST 3: Consistency check
    # ============================================================
    print(f"\n{'='*80}")
    print("TEST 3: CONSISTENCY CHECK - SINGLE vs BATCHED (DETERMINISTIC)")
    print(f"{'='*80}")

    test_prompt = prompts[0]
    config = CONFIGS["quality_cosine"]

    # Single inference
    single_inputs = tokenizer(
        [test_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2560,
        add_special_tokens=False
    ).to(device)

    print(f"\nSingle inference:")
    print(f"  input_ids shape: {single_inputs['input_ids'].shape}")
    print(f"  Valid tokens: {(single_inputs['attention_mask'] == 1).sum().item()}")

    with torch.inference_mode():
        single_output = model.generate_diffusion_quality(
            input_ids=single_inputs["input_ids"],
            attention_mask=single_inputs["attention_mask"],
            mask_token_id=mask_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            block_size=config["block_size"],
            steps=config["steps"],
            temperature=0.0,
            use_gumbel=False,
            gumbel_temperature=0.0,
            temperature_schedule=config.get("temperature_schedule", "constant"),
            final_temperature=config.get("final_temperature", 0.2),
            use_confidence_masking=config.get("use_confidence_masking", False),
        )

    single_gen = tokenizer.decode(single_output[0, single_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Batched inference with different prompts
    batch_inputs = tokenizer(
        [test_prompt, "Short", "A much longer prompt that is significantly different to force padding"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2560,
        add_special_tokens=False
    ).to(device)

    print(f"\nBatched inference:")
    print(f"  input_ids shape: {batch_inputs['input_ids'].shape}")
    for i in range(3):
        num_valid = (batch_inputs['attention_mask'][i] == 1).sum().item()
        print(f"  Sample {i} valid tokens: {num_valid}")

    with torch.inference_mode():
        batch_output = model.generate_diffusion_quality(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            mask_token_id=mask_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            block_size=config["block_size"],
            steps=config["steps"],
            temperature=0.0,
            use_gumbel=False,
            gumbel_temperature=0.0,
            temperature_schedule=config.get("temperature_schedule", "constant"),
            final_temperature=config.get("final_temperature", 0.2),
            use_confidence_masking=config.get("use_confidence_masking", False),
        )

    batch_gen = tokenizer.decode(batch_output[0, batch_inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\nPrompt: {test_prompt[:60]}...")
    print(f"\nSingle inference result:")
    print(f"  {single_gen[:100]}...")
    print(f"\nBatched inference result (same sample, index 0):")
    print(f"  {batch_gen[:100]}...")

    if single_gen == batch_gen:
        print(f"\n✅ PASS: Single and batched inference produce IDENTICAL results!")
    else:
        print(f"\n⚠️  Results differ!")
        print(f"   Single: {single_gen[:50]}")
        print(f"   Batch:  {batch_gen[:50]}")
        for i, (c1, c2) in enumerate(zip(single_gen, batch_gen)):
            if c1 != c2:
                print(f"   First difference at position {i}: '{c1}' vs '{c2}'")
                break

    print(f"\n{'='*80}")
    print("DEBUG COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()