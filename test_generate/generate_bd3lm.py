# generate_bd3lm.py
# Block Diffusion generation with KV cache support

import argparse
import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModel, AutoTokenizer, GenerationConfig


def top_k_logits(logits, k):
    """Apply top-k filtering to logits."""
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def top_p_logits(logits, p):
    """Apply top-p (nucleus) filtering to logits."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool),
        -1, sorted_indices, sorted_mask
    )
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    """Sample tokens from logits with temperature, top-k and top-p."""
    orig_shape = logits.shape[:-1]  # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
        
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    token_prob = torch.gather(probs, -1, token)

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    """Calculate number of tokens to transfer at each denoising step."""
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


def create_block_causal_mask(num_blocks, block_length, device):
    """
    Create block-causal attention mask.
    Each block can attend to itself and all previous blocks.
    """
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    attention_mask = block_mask.repeat_interleave(block_length, dim=0)\
                               .repeat_interleave(block_length, dim=1)
    return attention_mask


@torch.no_grad()
def block_diffusion_generate(
    model,
    prompt,
    mask_id,
    gen_length=128,
    block_length=8,
    denoising_steps=8,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    remasking_strategy='low_confidence_dynamic',
    confidence_threshold=0.85,
    eb_threshold=None,
    stopping_criteria_idx=None,
    use_kv_cache=True,
):
    """
    Generate text using block diffusion with KV cache support.
    
    Args:
        model: BD3LM model
        prompt: Tokenized prompt dict with 'input_ids'
        mask_id: Mask token ID
        gen_length: Number of tokens to generate
        block_length: Length of each block
        denoising_steps: Number of denoising steps per block
        temperature: Sampling temperature
        top_k: Top-K sampling parameter
        top_p: Top-P (nucleus) sampling parameter
        remasking_strategy: Strategy for selecting which tokens to unmask
        confidence_threshold: Threshold for low_confidence_dynamic strategy
        eb_threshold: Threshold for entropy_bounded strategy
        stopping_criteria_idx: Token IDs that stop generation
        use_kv_cache: Whether to use KV cache for efficiency
    
    Returns:
        Generated token IDs
    """
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = prompt['input_ids'].to(device)
    prompt_length = input_ids.shape[1]
    
    # Initialize KV cache
    past_key_values = DynamicCache() if use_kv_cache else None

    # Calculate total length (padded to block boundary)
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Create block-causal attention mask
    block_diffusion_attention_mask = create_block_causal_mask(
        num_blocks, block_length, device
    ).unsqueeze(0)  # [1, total_len, total_len]
    
    # Position IDs
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)

    # Initialize sequence with mask tokens
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids
    
    # Calculate prefill blocks
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    # ==================== Prefill Stage ====================
    if prefill_length > 0 and use_kv_cache:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        
        # Forward pass to populate KV cache
        model(
            input_ids=cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,  # Store KV in cache
        )

    # Number of tokens to transfer at each step
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    # ==================== Decode Stage ====================
    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length
        
        # Get current block
        cur_x = x[:, block_start:block_end].clone()
        
        # Attention mask for current block
        if use_kv_cache:
            # With KV cache: current block attends to all cached + current
            cached_len = past_key_values.get_seq_length() if past_key_values and len(past_key_values) > 0 else 0
            cur_attn_mask = block_diffusion_attention_mask[:, block_start:block_end, :block_end]
        else:
            cur_attn_mask = block_diffusion_attention_mask[:, block_start:block_end, :block_end]
        
        cur_position_ids = position_ids[:, block_start:block_end]
        
        # ==================== Denoising Loop ====================
        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            
            if mask_index.sum() == 0:
                # No more masks - store KV and move to next block
                if use_kv_cache:
                    model(
                        input_ids=cur_x,
                        attention_mask=cur_attn_mask,
                        position_ids=cur_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True,
                    )
                break

            # Forward pass for denoising (don't store KV yet)
            if use_kv_cache:
                outputs = model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,  # Don't store during denoising
                )
            else:
                # Without KV cache, need full context
                full_x = x[:, :block_end].clone()
                full_x[:, block_start:block_end] = cur_x
                full_attn_mask = block_diffusion_attention_mask[:, :block_end, :block_end]
                full_position_ids = position_ids[:, :block_end]
                
                outputs = model(
                    input_ids=full_x,
                    attention_mask=full_attn_mask,
                    position_ids=full_position_ids,
                    use_cache=False,
                )
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Get logits for current block only
            if not use_kv_cache:
                logits = logits[:, -block_length:]

            # Sample tokens
            x0, x0_p = sample_with_temperature_topk_topp(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # ==================== Remasking Strategy ====================
            if remasking_strategy == 'sequential':
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(cur_x.shape[0]):
                    if mask_index[j].any():
                        first_mask_idx = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                        transfer_index[j, first_mask_idx:first_mask_idx + num_transfer_tokens[step]] = True

            elif remasking_strategy == 'low_confidence_static':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                    transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum()
                    if num_high_confidence >= num_transfer_tokens[step]:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True

            elif remasking_strategy == "entropy_bounded":
                eps = 1e-12
                # Note: for entropy, we need the full probability distribution
                probs = F.softmax(logits, dim=-1)
                entropies = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
                entropies = torch.where(mask_index, entropies, torch.inf)
                ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                cumsum = torch.cumsum(ent_sorted, dim=1)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(probs.shape[0]):
                    k = torch.searchsorted(
                        cumsum[j], 
                        torch.tensor(eb_threshold, device=device), 
                        right=False
                    ).item()
                    k = max(1, min(k, int(mask_index[j].sum().item())))
                    selected_indices = order[j, :k]
                    transfer_index[j, selected_indices] = True
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

            # Update current block with sampled tokens
            cur_x[transfer_index] = x0[transfer_index]

        # Update full sequence
        x[:, block_start:block_end] = cur_x
        
        # After denoising is done, store the final KV if using cache
        if use_kv_cache and mask_index.sum() > 0:
            model(
                input_ids=cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )
        
        # Check stopping criteria
        if stopping_criteria_idx is not None:
            generated = x[:, prompt_length:block_end]
            if any(stop_idx in generated for stop_idx in stopping_criteria_idx):
                break

    return x


@torch.no_grad()
def block_diffusion_generate_simple(
    model,
    prompt,
    mask_id,
    gen_length=128,
    block_length=8,
    denoising_steps=8,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    stopping_criteria_idx=None,
):
    """
    Simplified generation without KV cache (for comparison/debugging).
    """
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = prompt['input_ids'].to(device)
    prompt_length = input_ids.shape[1]
    
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Create attention mask
    block_diffusion_attention_mask = create_block_causal_mask(
        num_blocks, block_length, device
    ).unsqueeze(0)
    
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)

    # Initialize with masks
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids
    
    prefill_blocks = prompt_length // block_length
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length
        
        for step in range(denoising_steps + 1):
            cur_x = x[:, block_start:block_end]
            mask_index = (cur_x == mask_id)
            
            if mask_index.sum() == 0:
                break

            # Full forward pass
            full_x = x[:, :block_end]
            full_attn_mask = block_diffusion_attention_mask[:, :block_end, :block_end]
            full_position_ids = position_ids[:, :block_end]
            
            outputs = model(
                input_ids=full_x,
                attention_mask=full_attn_mask,
                position_ids=full_position_ids,
                use_cache=False,
            )
            
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            logits = logits[:, -block_length:]

            x0, x0_p = sample_with_temperature_topk_topp(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # Low confidence dynamic remasking
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, idx = torch.topk(confidence[j], min(num_transfer_tokens[step], mask_index[j].sum()))
                transfer_index[j, idx] = True

            x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        
        if stopping_criteria_idx is not None:
            generated = x[:, prompt_length:block_end]
            if any(stop_idx in generated for stop_idx in stopping_criteria_idx):
                break

    return x


def parse_args():
    parser = argparse.ArgumentParser(description="BD3LM Block Diffusion Generation")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pretrained model directory")
    parser.add_argument("--trust_remote_code", action='store_true')
    parser.add_argument("--mask_id", type=int, default=None,
                        help="Mask token id for Diffusion")
    parser.add_argument("--prompt_length", type=int, default=4096,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--gen_length", type=int, default=2048,
                        help="Maximum generation length in tokens")
    parser.add_argument("--block_length", type=int, default=8,
                        help="Length of token block")
    parser.add_argument("--denoising_steps", type=int, default=8,
                        help="Number of denoising steps per block")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-K sampling (0 to disable)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-P sampling probability threshold")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        choices=["low_confidence_dynamic", "low_confidence_static",
                                 "sequential", "entropy_bounded"],
                        help="Strategy for remasking tokens")
    parser.add_argument("--confidence_threshold", type=float, default=0.85,
                        help="Confidence threshold for low-confidence remasking")
    parser.add_argument("--eb_threshold", type=float, default=0.35,
                        help="Entropy threshold for entropy bounded sampling")
    parser.add_argument("--stopping_criteria_idx", type=int, nargs="+", default=None,
                        help="List of token IDs that stop generation")
    parser.add_argument("--use_kv_cache", action='store_true', default=True,
                        help="Use KV cache for efficient generation")
    parser.add_argument("--no_kv_cache", action='store_true',
                        help="Disable KV cache")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    
    args = parser.parse_args()
    
    # Validation
    if args.remasking_strategy == "low_confidence_dynamic" and args.confidence_threshold is None:
        parser.error("--confidence_threshold required for low_confidence_dynamic")
    if args.remasking_strategy == "entropy_bounded" and args.eb_threshold is None:
        parser.error("--eb_threshold required for entropy_bounded")
    
    if args.no_kv_cache:
        args.use_kv_cache = False
        
    return args


if __name__ == "__main__":
    args = parse_args()

    print("Loading model...")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype_map[args.dtype],
        device_map=args.device
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
    )

    # Get mask token ID
    if args.mask_id is None:
        if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token:
            args.mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        else:
            args.mask_id = model.config.mask_token_id if hasattr(model.config, 'mask_token_id') else model.config.vocab_size - 1
    
    # Get stopping criteria
    if args.stopping_criteria_idx is None:
        try:
            gen_cfg = GenerationConfig.from_pretrained(args.model_dir)
            args.stopping_criteria_idx = gen_cfg.eos_token_id
        except:
            args.stopping_criteria_idx = tokenizer.eos_token_id
            
    if isinstance(args.stopping_criteria_idx, int):
        args.stopping_criteria_idx = [args.stopping_criteria_idx]
        
    print(f"Configuration: {args}")

    # ==================== 修改部分 ====================
    # 1. 直接使用字符串作为 prompt
    origin_prompt = "I like science," 

    # 2. 直接对字符串进行 encode，不使用 chat_template
    tokens = tokenizer(
        origin_prompt,
        return_tensors='pt',
        padding=True,
        truncation=True,
        add_special_tokens=True, # Base模型通常建议保留BOS等特殊符号
        max_length=args.prompt_length
    )
    # =================================================

    print(f"Prompt length: {tokens['input_ids'].shape[1]} tokens")
    print(f"Using KV cache: {args.use_kv_cache}")

    # Generate
    output_ids = block_diffusion_generate(
        model,
        prompt=tokens,
        mask_id=args.mask_id,
        gen_length=args.gen_length,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        eb_threshold=args.eb_threshold,
        stopping_criteria_idx=args.stopping_criteria_idx,
        use_kv_cache=args.use_kv_cache,
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    
    # Clean up mask tokens in output
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token:
        cleaned_text = output_text.replace(tokenizer.mask_token, '')
    else:
        cleaned_text = output_text.replace('<|MASK|>', '')
    
    print("\n" + "="*50)
    print("Generated Output:")
    print("="*50)
    print(cleaned_text)