"""
debug.py
Diffusion Generation Debug Script - Aligned with New Modeling File
- Uses unpadding for prefill (no explicit position_ids needed)
- KV cache stores from position 0 (no cache_leftpad)
- Tests both single sample and batched inference
"""

import torch
import torch.nn.functional as F
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# MAIN DEBUG FUNCTION
# ============================================================================

def main():
    DIFFUSION_MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_main_exp/checkpoint-77335"
    HELLASWAG_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/Rowan/hellaswag/main"

    device = "cuda:0"
    MAX_NEW_TOKENS = 32
    NUM_SAMPLES = 5

    CONFIGS = {
        "block_kvcache": {
            "method": "block_kvcache",
            "block_size": 16,
            "steps": 16,
            "temperature": 0.7,
        },
        "quality": {
            "method": "quality",
            "block_size": 16,
            "steps": 16,
            "gumbel_temperature": 0.7,
            "temperature_schedule": "cosine",
            "final_temperature": 0.2,
        },
    }

    # Load model
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

    mask_token_id = model.config.mask_token_id
    print(f"Model loaded successfully!")
    print(f"mask_token_id: {mask_token_id}")
    print(f"Model type: {type(model).__name__}")

    # Check if model has the generation methods
    has_block_kvcache = hasattr(model, 'generate_diffusion_block_kvcache')
    has_quality = hasattr(model, 'generate_diffusion_quality')
    print(f"Has generate_diffusion_block_kvcache: {has_block_kvcache}")
    print(f"Has generate_diffusion_quality: {has_quality}")

    if not has_block_kvcache or not has_quality:
        print("ERROR: Model doesn't have required generation methods!")
        print("Make sure you're using the updated modeling file.")
        return

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(HELLASWAG_PATH)
    prompts = list(dataset["validation"]["ctx"][:NUM_SAMPLES])
    print(f"Loaded {len(prompts)} prompts")

    # ============================================================================
    # TEST 1: Single sample inference (baseline)
    # ============================================================================
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
        print(f"attention_mask sum: {inputs['attention_mask'].sum().item()}")

        # Compute valid lengths for reference
        valid_lengths = inputs['attention_mask'].sum(dim=1)
        left_pad = (inputs['attention_mask'] == 0).sum(dim=1)
        print(f"valid_lengths: {valid_lengths.tolist()}")
        print(f"left_pad: {left_pad.tolist()}")

        prompt_len = inputs["input_ids"].shape[1]

        for config_name, config in CONFIGS.items():
            with torch.inference_mode():
                if config["method"] == "block_kvcache":
                    output_ids = model.generate_diffusion_block_kvcache(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        mask_token_id=mask_token_id,
                        max_new_tokens=MAX_NEW_TOKENS,
                        block_size=config["block_size"],
                        steps=config["steps"],
                        temperature=config["temperature"],
                    )
                elif config["method"] == "quality":
                    output_ids = model.generate_diffusion_quality(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        mask_token_id=mask_token_id,
                        max_new_tokens=MAX_NEW_TOKENS,
                        block_size=config["block_size"],
                        steps=config["steps"],
                        temperature=1.0,
                        use_gumbel=True,
                        gumbel_temperature=config["gumbel_temperature"],
                        temperature_schedule=config.get("temperature_schedule", "constant"),
                        final_temperature=config.get("final_temperature", 0.2),
                    )

            generated_ids = output_ids[:, prompt_len:]
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"  {config_name}: {decoded[:100]}...")

    # ============================================================================
    # TEST 2: Batched inference with padding (the key test)
    # ============================================================================
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
    valid_lengths = inputs['attention_mask'].sum(dim=1)
    left_pad = (inputs['attention_mask'] == 0).sum(dim=1)

    print(f"\nPadding analysis:")
    for i in range(len(batch_prompts)):
        num_pad = left_pad[i].item()
        num_valid = valid_lengths[i].item()
        print(f"  Sample {i}: left_pad={num_pad}, valid={num_valid}, total={num_pad + num_valid}")

    prompt_len = inputs["input_ids"].shape[1]

    for config_name, config in CONFIGS.items():
        print(f"\n--- {config_name.upper()} (BATCHED) ---")

        with torch.inference_mode():
            if config["method"] == "block_kvcache":
                output_ids = model.generate_diffusion_block_kvcache(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    mask_token_id=mask_token_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    block_size=config["block_size"],
                    steps=config["steps"],
                    temperature=config["temperature"],
                )
            elif config["method"] == "quality":
                output_ids = model.generate_diffusion_quality(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    mask_token_id=mask_token_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    block_size=config["block_size"],
                    steps=config["steps"],
                    temperature=1.0,
                    use_gumbel=True,
                    gumbel_temperature=config["gumbel_temperature"],
                    temperature_schedule=config.get("temperature_schedule", "constant"),
                    final_temperature=config.get("final_temperature", 0.2),
                )

        generated_ids = output_ids[:, prompt_len:]
        generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for i, gen in enumerate(generations):
            print(f"  Sample {i}: {gen[:80]}...")

    # ============================================================================
    # TEST 3: Consistency check - Single vs Batched (deterministic)
    # ============================================================================
    print(f"\n{'='*80}")
    print("TEST 3: CONSISTENCY CHECK - SINGLE vs BATCHED (DETERMINISTIC)")
    print(f"{'='*80}")

    test_prompt = prompts[0]

    # Single inference
    single_inputs = tokenizer(
        [test_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2560,
        add_special_tokens=False
    ).to(device)

    single_valid = single_inputs['attention_mask'].sum(dim=1)
    single_leftpad = (single_inputs['attention_mask'] == 0).sum(dim=1)

    print(f"\nSingle inference:")
    print(f"  input_ids shape: {single_inputs['input_ids'].shape}")
    print(f"  left_pad: {single_leftpad.tolist()}")
    print(f"  valid_lengths: {single_valid.tolist()}")

    with torch.inference_mode():
        single_output = model.generate_diffusion_block_kvcache(
            input_ids=single_inputs["input_ids"],
            attention_mask=single_inputs["attention_mask"],
            mask_token_id=mask_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            block_size=16,
            steps=16,
            temperature=0.0,  # Deterministic (argmax)
        )

    single_gen = tokenizer.decode(
        single_output[0, single_inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )

    # Batched inference with padding from other samples
    batch_inputs = tokenizer(
        [test_prompt, "Short", "A much longer prompt that is significantly different to force padding"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2560,
        add_special_tokens=False
    ).to(device)

    batch_valid = batch_inputs['attention_mask'].sum(dim=1)
    batch_leftpad = (batch_inputs['attention_mask'] == 0).sum(dim=1)

    print(f"\nBatched inference:")
    print(f"  input_ids shape: {batch_inputs['input_ids'].shape}")
    print(f"  left_pad: {batch_leftpad.tolist()}")
    print(f"  valid_lengths: {batch_valid.tolist()}")

    with torch.inference_mode():
        batch_output = model.generate_diffusion_block_kvcache(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            mask_token_id=mask_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            block_size=16,
            steps=16,
            temperature=0.0,  # Deterministic (argmax)
        )

    batch_gen = tokenizer.decode(
        batch_output[0, batch_inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )

    print(f"\nPrompt: {test_prompt[:60]}...")
    print(f"\nSingle inference result:")
    print(f"  {single_gen[:100]}...")
    print(f"\nBatched inference result (same sample, index 0):")
    print(f"  {batch_gen[:100]}...")

    # Check if they match
    if single_gen == batch_gen:
        print(f"\n✅ PASS: Single and batched inference produce IDENTICAL results!")
        print(f"   Left-padding handling is working correctly!")
    else:
        print(f"\n❌ FAIL: Results differ!")
        print(f"   Single: {single_gen[:50]}")
        print(f"   Batch:  {batch_gen[:50]}")
        # Find first difference
        min_len = min(len(single_gen), len(batch_gen))
        for i in range(min_len):
            if single_gen[i] != batch_gen[i]:
                print(f"   First difference at position {i}: '{single_gen[i]}' vs '{batch_gen[i]}'")
                ctx_start = max(0, i - 10)
                print(f"   Context: ...{single_gen[ctx_start:i+10]}... vs ...{batch_gen[ctx_start:i+10]}...")
                break
        else:
            if len(single_gen) != len(batch_gen):
                print(f"   Length difference: {len(single_gen)} vs {len(batch_gen)}")

    # ============================================================================
    # TEST 4: Verify unpadding works correctly
    # ============================================================================
    print(f"\n{'='*80}")
    print("TEST 4: VERIFY UNPADDING BEHAVIOR")
    print(f"{'='*80}")

    # Create a batch with known padding patterns
    test_prompts_varied = [
        "A",  # Very short - will have most padding
        "A B C D E F G H I J",  # Medium
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",  # Long
    ]

    test_inputs = tokenizer(
        test_prompts_varied,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2560,
        add_special_tokens=False
    ).to(device)

    test_valid = test_inputs['attention_mask'].sum(dim=1)
    test_leftpad = (test_inputs['attention_mask'] == 0).sum(dim=1)

    print(f"\nTest prompts with varying lengths:")
    all_checks_passed = True

    for i, prompt in enumerate(test_prompts_varied):
        total_len = test_inputs['input_ids'].shape[1]
        valid_len = test_valid[i].item()
        pad_len = test_leftpad[i].item()

        print(f"\n  Sample {i}: '{prompt[:30]}...'")
        print(f"    Total length: {total_len}, Valid: {valid_len}, Padding: {pad_len}")

        # Check 1: padding + valid should equal total
        check1 = pad_len + valid_len == total_len
        print(f"    Check 1 - pad + valid == total: {'✓' if check1 else '✗'}")
        all_checks_passed &= check1

        # Check 2: first `pad_len` tokens should be pad_token_id
        if pad_len > 0:
            pad_tokens = test_inputs['input_ids'][i, :pad_len]
            check2 = (pad_tokens == tokenizer.pad_token_id).all().item()
            print(f"    Check 2 - First {pad_len} tokens are pad_token_id: {'✓' if check2 else '✗'}")
            all_checks_passed &= check2

        # Check 3: valid tokens start after padding
        if pad_len > 0:
            first_valid = test_inputs['input_ids'][i, pad_len].item()
            first_valid_mask = test_inputs['attention_mask'][i, pad_len].item()
            check3 = first_valid_mask == 1
            print(f"    Check 3 - First valid token has mask=1: {'✓' if check3 else '✗'}")
            all_checks_passed &= check3

    if all_checks_passed:
        print(f"\n✅ All unpadding checks passed!")
    else:
        print(f"\n❌ Some unpadding checks failed!")

    # ============================================================================
    # TEST 5: Test KV cache state tracking
    # ============================================================================
    print(f"\n{'='*80}")
    print("TEST 5: KV CACHE STATE TRACKING")
    print(f"{'='*80}")

    # Use a simple batch
    debug_prompts = ["Hello world", "Short"]
    debug_inputs = tokenizer(
        debug_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2560,
        add_special_tokens=False
    ).to(device)

    debug_valid = debug_inputs['attention_mask'].sum(dim=1)
    debug_leftpad = (debug_inputs['attention_mask'] == 0).sum(dim=1)

    print(f"\nDebug batch:")
    print(f"  Prompts: {debug_prompts}")
    print(f"  input_ids shape: {debug_inputs['input_ids'].shape}")
    print(f"  left_pad: {debug_leftpad.tolist()}")
    print(f"  valid_lengths: {debug_valid.tolist()}")

    # Initialize cache and show state
    batch_size = len(debug_prompts)
    prefix_len = debug_inputs['input_ids'].shape[1]
    max_new = 8

    cache = model._init_kv_cache(batch_size, prefix_len + max_new, device, torch.bfloat16)

    print(f"\n  Initial cache state:")
    print(f"    cache_seqlens: {cache.get_seq_lens().tolist()}")

    # Simulate prefill - set seq_lens to valid lengths
    cache.set_seq_lens(debug_valid)
    print(f"\n  After prefill (set_seq_lens to valid_lengths):")
    print(f"    cache_seqlens: {cache.get_seq_lens().tolist()}")

    # Simulate generation steps
    print(f"\n  Simulated generation (block_size=2):")
    block_size = 2
    for step in range(3):
        # Rewind to process block
        rewind_lens = (debug_valid + step * block_size - 1).clamp(min=0)
        cache.set_seq_lens(rewind_lens)
        print(f"    Step {step} rewind: cache_seqlens = {cache.get_seq_lens().tolist()}")

        # After processing block
        new_lens = debug_valid + (step + 1) * block_size
        cache.set_seq_lens(new_lens)
        print(f"    Step {step} after block: cache_seqlens = {cache.get_seq_lens().tolist()}")

    # ============================================================================
    # TEST 6: Compare prefill output with and without batch padding
    # ============================================================================
    print(f"\n{'='*80}")
    print("TEST 6: PREFILL OUTPUT COMPARISON")
    print(f"{'='*80}")

    test_prompt_single = "The quick brown fox jumps over the lazy dog"

    # Single (no padding)
    single_input = tokenizer(
        [test_prompt_single],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2560,
        add_special_tokens=False
    ).to(device)

    # Batch with padding
    batch_input = tokenizer(
        [test_prompt_single, "Short"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2560,
        add_special_tokens=False
    ).to(device)

    print(f"\nSingle input (no padding):")
    print(f"  Shape: {single_input['input_ids'].shape}")
    print(f"  Tokens: {single_input['input_ids'][0, :10].tolist()}...")

    print(f"\nBatch input (with padding):")
    print(f"  Shape: {batch_input['input_ids'].shape}")
    print(f"  Sample 0 left_pad: {(batch_input['attention_mask'][0] == 0).sum().item()}")
    print(f"  Sample 0 valid: {batch_input['attention_mask'][0].sum().item()}")

    # Run forward pass to check hidden states
    with torch.inference_mode():
        single_out = model.model(
            input_ids=single_input['input_ids'],
            attention_mask=single_input['attention_mask'],
            causal=True,
            return_dict=True
        )

        batch_out = model.model(
            input_ids=batch_input['input_ids'],
            attention_mask=batch_input['attention_mask'],
            causal=True,
            return_dict=True
        )

    # Compare the hidden states for the test prompt
    single_hidden = single_out.last_hidden_state[0]  # (seq_len, hidden)

    # For batch, we need to extract the valid portion (skip left padding)
    batch_leftpad_0 = (batch_input['attention_mask'][0] == 0).sum().item()
    batch_hidden = batch_out.last_hidden_state[0, batch_leftpad_0:]  # (valid_len, hidden)

    print(f"\nHidden state comparison:")
    print(f"  Single hidden shape: {single_hidden.shape}")
    print(f"  Batch hidden shape (after extracting valid): {batch_hidden.shape}")

    if single_hidden.shape == batch_hidden.shape:
        diff = (single_hidden - batch_hidden).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-4:
            print(f"  ✅ Hidden states match! Unpadding is working correctly.")
        else:
            print(f"  ⚠️ Hidden states differ. This may indicate RoPE position issues.")
    else:
        print(f"  ❌ Shape mismatch!")

    # ============================================================================
    # TEST 7: Full generation with different batch compositions
    # ============================================================================
    print(f"\n{'='*80}")
    print("TEST 7: GENERATION WITH VARIOUS BATCH COMPOSITIONS")
    print(f"{'='*80}")

    reference_prompt = prompts[0]

    # Generate reference output (single)
    ref_input = tokenizer(
        [reference_prompt],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2560,
        add_special_tokens=False
    ).to(device)

    with torch.inference_mode():
        ref_output = model.generate_diffusion_block_kvcache(
            input_ids=ref_input["input_ids"],
            attention_mask=ref_input["attention_mask"],
            mask_token_id=mask_token_id,
            max_new_tokens=16,
            block_size=8,
            steps=8,
            temperature=0.0,
        )

    ref_gen = tokenizer.decode(ref_output[0, ref_input["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nReference (single, no padding):")
    print(f"  {ref_gen[:80]}...")

    # Test with different batch compositions
    batch_compositions = [
        [reference_prompt, "Short"],  # Short padding
        [reference_prompt, "A" * 100],  # Long padding (if reference is shorter)
        ["Tiny", reference_prompt, "Medium length text here"],  # Middle position
    ]

    for comp_idx, composition in enumerate(batch_compositions):
        # Find position of reference prompt
        ref_pos = composition.index(reference_prompt)

        batch_in = tokenizer(
            composition,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2560,
            add_special_tokens=False
        ).to(device)

        with torch.inference_mode():
            batch_out = model.generate_diffusion_block_kvcache(
                input_ids=batch_in["input_ids"],
                attention_mask=batch_in["attention_mask"],
                mask_token_id=mask_token_id,
                max_new_tokens=16,
                block_size=8,
                steps=8,
                temperature=0.0,
            )

        batch_gen = tokenizer.decode(
            batch_out[ref_pos, batch_in["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )

        match = ref_gen == batch_gen
        print(f"\nComposition {comp_idx + 1} (ref at position {ref_pos}):")
        print(f"  Batch sizes: {[len(tokenizer.encode(p)) for p in composition]}")
        print(f"  Result: {batch_gen[:80]}...")
        print(f"  Match: {'✅' if match else '❌'}")

    print(f"\n{'='*80}")
    print("DEBUG COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()