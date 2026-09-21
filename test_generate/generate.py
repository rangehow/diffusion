# generate_bd3lm.py
# Causal Block Diffusion generation with KV cache support for ModernBERT

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


def create_causal_mask(
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    dtype: torch.dtype,
    past_length: int = 0,
) -> torch.Tensor:
    """
    Create a causal attention mask for standard autoregressive attention.
    
    Each position i in the query can attend to positions 0 to past_length + i in the key.
    
    Args:
        batch_size: Batch size
        query_length: Length of query sequence
        key_length: Total length of key sequence (past + current)
        device: Device
        dtype: Data type
        past_length: Length of past/cached sequence
        
    Returns:
        Attention mask of shape [batch_size, 1, query_length, key_length]
        where allowed positions have 0 and masked positions have -inf
    """
    # Query positions in the full sequence: [past_length, past_length + query_length)
    query_positions = torch.arange(past_length, past_length + query_length, device=device).unsqueeze(1)
    # Key positions: [0, key_length)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    
    # Causal mask: query can attend to key if query_pos >= key_pos
    causal_mask = query_positions >= key_positions  # [query_length, key_length]
    
    # Convert to attention mask format (0 for attend, -inf for mask)
    attention_mask = torch.zeros((batch_size, 1, query_length, key_length), device=device, dtype=dtype)
    attention_mask = attention_mask.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    return attention_mask


@torch.no_grad()
def causal_block_diffusion_generate(
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
    verbose=False,
):
    """
    Generate text using causal block diffusion with KV cache support.
    
    This implements block diffusion for a standard causal (unidirectional) language model:
    1. Global causal mask: token i only sees tokens 0 to i-1
    2. Block-wise masking: append a block of mask_tokens to the sequence
    3. Iterative denoising: for each block, run multiple denoising steps
    4. KV cache handling: only commit to cache after block finalization
    
    Args:
        model: Language model with KV cache support
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
        verbose: Print progress information
    
    Returns:
        Generated token IDs
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    input_ids = prompt['input_ids'].to(device)
    batch_size = input_ids.shape[0]
    prompt_length = input_ids.shape[1]
    
    # Calculate total length
    num_blocks = (gen_length + block_length - 1) // block_length
    total_gen_length = num_blocks * block_length
    total_length = prompt_length + total_gen_length

    if verbose:
        print(f"Prompt length: {prompt_length}")
        print(f"Generation length: {total_gen_length} ({num_blocks} blocks of {block_length})")
        print(f"Total length: {total_length}")
        print(f"Denoising steps per block: {denoising_steps}")
        print(f"Using KV cache: {use_kv_cache}")

    # Initialize sequence with mask tokens
    x = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids
    
    # Initialize KV cache
    past_key_values = DynamicCache() if use_kv_cache else None

    # Number of tokens to transfer at each step
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    # ==================== Prefill Stage ====================
    # Process the prompt to populate KV cache
    if prompt_length > 0:
        if verbose:
            print(f"\nPrefilling prompt ({prompt_length} tokens)...")
        
        prefill_ids = x[:, :prompt_length]
        
        # Create causal mask for prefill
        prefill_mask = create_causal_mask(
            batch_size=batch_size,
            query_length=prompt_length,
            key_length=prompt_length,
            device=device,
            dtype=dtype,
            past_length=0,
        )
        
        # Position IDs for prefill
        prefill_position_ids = torch.arange(prompt_length, device=device).unsqueeze(0).expand(batch_size, -1)
        cache_position = torch.arange(prompt_length, device=device)
        
        # Forward pass to populate KV cache
        if use_kv_cache:
            outputs = model(
                input_ids=prefill_ids,
                attention_mask=prefill_mask,
                position_ids=prefill_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                causal=True,
            )
            past_key_values = outputs.past_key_values

    # ==================== Decode Stage ====================
    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        block_end = block_start + block_length
        
        if verbose:
            print(f"\nGenerating block {block_idx + 1}/{num_blocks} (positions {block_start}-{block_end})")
        
        # Current block tokens (initially all masks)
        cur_block = x[:, block_start:block_end].clone()
        
        # Get current cache length (committed KV states)
        cached_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        # Save cache state before denoising (for restoration between iterations)
        # We need to create a snapshot of the cache since we don't want denoising iterations
        # to modify the committed cache
        if use_kv_cache:
            # Store the current cache lengths for each layer
            cache_snapshot_lens = [
                past_key_values.key_cache[i].shape[2] if i < len(past_key_values.key_cache) else 0
                for i in range(len(past_key_values.key_cache)) if past_key_values.key_cache
            ] if past_key_values.key_cache else []
        
        # ==================== Denoising Loop ====================
        for step in range(denoising_steps + 1):
            mask_index = (cur_block == mask_id)
            num_masks = mask_index.sum().item()
            
            if num_masks == 0:
                if verbose:
                    print(f"  Step {step}: No masks remaining, block complete")
                break
            
            if verbose:
                print(f"  Step {step}: {num_masks} masks remaining")
            
            # Position IDs for current block
            block_position_ids = torch.arange(block_start, block_end, device=device).unsqueeze(0).expand(batch_size, -1)
            cache_position = torch.arange(block_start, block_end, device=device)
            
            # Create causal mask for current block attending to cached + current
            # Key length = cached_len + block_length
            total_kv_len = cached_len + block_length
            block_mask = create_causal_mask(
                batch_size=batch_size,
                query_length=block_length,
                key_length=total_kv_len,
                device=device,
                dtype=dtype,
                past_length=cached_len,
            )
            
            # Forward pass (without committing to cache during denoising)
            # We use use_cache=False to avoid modifying the cache
            if use_kv_cache:
                # Create a temporary cache by copying the committed cache
                # This is a shallow approach - we just don't commit new KV states
                outputs = model(
                    input_ids=cur_block,
                    attention_mask=block_mask,
                    position_ids=block_position_ids,
                    past_key_values=past_key_values,
                    use_cache=False,  # Don't update cache during denoising
                    cache_position=cache_position,
                    causal=True,
                )
            else:
                # Without KV cache, need full context each time
                full_x = x[:, :block_end].clone()
                full_x[:, block_start:block_end] = cur_block
                
                full_mask = create_causal_mask(
                    batch_size=batch_size,
                    query_length=block_end,
                    key_length=block_end,
                    device=device,
                    dtype=dtype,
                    past_length=0,
                )
                
                full_position_ids = torch.arange(block_end, device=device).unsqueeze(0).expand(batch_size, -1)
                
                outputs = model(
                    input_ids=full_x,
                    attention_mask=full_mask,
                    position_ids=full_position_ids,
                    use_cache=False,
                    causal=True,
                )
            
            logits = outputs.logits
            
            # Get logits for current block only (if using full context)
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
                # Transfer tokens left-to-right
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch_size):
                    if mask_index[j].any():
                        first_mask_idx = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                        end_idx = min(first_mask_idx + num_transfer_tokens[step].item(), block_length)
                        transfer_index[j, first_mask_idx:end_idx] = True

            elif remasking_strategy == 'low_confidence_static':
                # Transfer top-k confident tokens
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch_size):
                    num_to_transfer = min(num_transfer_tokens[step].item(), mask_index[j].sum().item())
                    if num_to_transfer > 0:
                        _, idx = torch.topk(confidence[j], num_to_transfer)
                        transfer_index[j, idx] = True

            elif remasking_strategy == 'low_confidence_dynamic':
                # Transfer high-confidence tokens, or top-k if not enough
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch_size):
                    high_conf_mask = confidence[j] > confidence_threshold
                    num_high_confidence = high_conf_mask.sum().item()
                    target_transfer = min(num_transfer_tokens[step].item(), mask_index[j].sum().item())
                    
                    if num_high_confidence >= target_transfer:
                        # Keep all high confidence tokens
                        transfer_index[j] = high_conf_mask
                    else:
                        # Take top-k by confidence
                        if target_transfer > 0:
                            _, idx = torch.topk(confidence[j], target_transfer)
                            transfer_index[j, idx] = True

            elif remasking_strategy == "entropy_bounded":
                # Transfer tokens with low entropy
                eps = 1e-12
                probs = F.softmax(logits, dim=-1)
                entropies = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
                entropies = torch.where(mask_index, entropies, torch.tensor(float('inf'), device=device))
                
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(batch_size):
                    ent_sorted, order = torch.sort(entropies[j], descending=False)
                    cumsum = torch.cumsum(ent_sorted, dim=0)
                    k = torch.searchsorted(
                        cumsum, 
                        torch.tensor(eb_threshold, device=device), 
                        right=False
                    ).item()
                    k = max(1, min(k, int(mask_index[j].sum().item())))
                    selected_indices = order[:k]
                    transfer_index[j, selected_indices] = True
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

            # Update current block with sampled tokens (only at transfer positions)
            cur_block[transfer_index] = x0[transfer_index]

        # ==================== Block Finalization ====================
        # Update the full sequence
        x[:, block_start:block_end] = cur_block
        
        # Commit block to KV cache after finalization
        if use_kv_cache:
            if verbose:
                print(f"  Committing block to KV cache...")
            
            # Final forward pass with the finalized block to update cache
            block_position_ids = torch.arange(block_start, block_end, device=device).unsqueeze(0).expand(batch_size, -1)
            cache_position = torch.arange(block_start, block_end, device=device)
            
            total_kv_len = cached_len + block_length
            block_mask = create_causal_mask(
                batch_size=batch_size,
                query_length=block_length,
                key_length=total_kv_len,
                device=device,
                dtype=dtype,
                past_length=cached_len,
            )
            
            outputs = model(
                input_ids=cur_block,
                attention_mask=block_mask,
                position_ids=block_position_ids,
                past_key_values=past_key_values,
                use_cache=True,  # Now commit to cache
                cache_position=cache_position,
                causal=True,
            )
            past_key_values = outputs.past_key_values
        
        # Check stopping criteria
        if stopping_criteria_idx is not None:
            generated = x[:, prompt_length:block_end]
            for stop_idx in stopping_criteria_idx:
                if (generated == stop_idx).any():
                    if verbose:
                        print(f"Stopping criteria met (token {stop_idx})")
                    return x[:, :block_end]

    return x


@torch.no_grad()
def causal_block_diffusion_generate_simple(
    model,
    prompt,
    mask_id,
    gen_length=128,
    block_length=8,
    denoising_steps=8,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    confidence_threshold=0.85,
    stopping_criteria_idx=None,
    verbose=False,
):
    """
    Simplified causal block diffusion generation without KV cache.
    Useful for debugging or comparison.
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    input_ids = prompt['input_ids'].to(device)
    batch_size = input_ids.shape[0]
    prompt_length = input_ids.shape[1]
    
    num_blocks = (gen_length + block_length - 1) // block_length
    total_length = prompt_length + num_blocks * block_length

    # Initialize with masks
    x = torch.full((batch_size, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids
    
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

    for block_idx in range(num_blocks):
        block_start = prompt_length + block_idx * block_length
        block_end = block_start + block_length
        
        if verbose:
            print(f"Block {block_idx + 1}/{num_blocks}")
        
        for step in range(denoising_steps + 1):
            cur_block = x[:, block_start:block_end]
            mask_index = (cur_block == mask_id)
            
            if mask_index.sum() == 0:
                break

            # Full forward pass with causal mask
            full_x = x[:, :block_end]
            full_mask = create_causal_mask(
                batch_size=batch_size,
                query_length=block_end,
                key_length=block_end,
                device=device,
                dtype=dtype,
            )
            full_position_ids = torch.arange(block_end, device=device).unsqueeze(0).expand(batch_size, -1)
            
            outputs = model(
                input_ids=full_x,
                attention_mask=full_mask,
                position_ids=full_position_ids,
                use_cache=False,
                causal=True,
            )
            
            logits = outputs.logits[:, -block_length:]

            x0, x0_p = sample_with_temperature_topk_topp(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # Low confidence dynamic remasking
            confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(batch_size):
                high_conf = confidence[j] > confidence_threshold
                target = min(num_transfer_tokens[step].item(), mask_index[j].sum().item())
                if high_conf.sum() >= target:
                    transfer_index[j] = high_conf
                else:
                    if target > 0:
                        _, idx = torch.topk(confidence[j], target)
                        transfer_index[j, idx] = True

            x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        
        if stopping_criteria_idx is not None:
            generated = x[:, prompt_length:block_end]
            for stop_idx in stopping_criteria_idx:
                if (generated == stop_idx).any():
                    return x[:, :block_end]

    return x


def parse_args():
    parser = argparse.ArgumentParser(description="Causal Block Diffusion Generation")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the pretrained model directory")
    parser.add_argument("--trust_remote_code", action='store_true')
    parser.add_argument("--mask_id", type=int, default=None,
                        help="Mask token id for Diffusion")
    parser.add_argument("--prompt", type=str, default="I like science,",
                        help="Input prompt for generation")
    parser.add_argument("--prompt_length", type=int, default=4096,
                        help="Maximum prompt length in tokens")
    parser.add_argument("--gen_length", type=int, default=128,
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
    parser.add_argument("--verbose", action='store_true',
                        help="Print detailed progress")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"],
                        help="Attention implementation to use (sdpa recommended for KV cache)")
    
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

    print("=" * 60)
    print("Causal Block Diffusion Generation")
    print("=" * 60)
    
    print("\nLoading model...")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
    )

    # Load model with specified attention implementation
    model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device,
        attn_implementation=args.attn_implementation,
    )

    # Get mask token ID
    if args.mask_id is None:
        if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token:
            args.mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        elif hasattr(model.config, 'mask_token_id'):
            args.mask_id = model.config.mask_token_id
        else:
            args.mask_id = model.config.vocab_size - 1
            print(f"Warning: Using vocab_size - 1 ({args.mask_id}) as mask_id")

    # Get stopping criteria
    if args.stopping_criteria_idx is None:
        try:
            gen_cfg = GenerationConfig.from_pretrained(args.model_dir)
            args.stopping_criteria_idx = gen_cfg.eos_token_id
        except:
            args.stopping_criteria_idx = tokenizer.eos_token_id
            
    if isinstance(args.stopping_criteria_idx, int):
        args.stopping_criteria_idx = [args.stopping_criteria_idx]
    
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_dir}")
    print(f"  Attention implementation: {args.attn_implementation}")
    print(f"  Device: {args.device}")
    print(f"  Data type: {args.dtype}")
    print(f"  Mask token ID: {args.mask_id}")
    print(f"  Block length: {args.block_length}")
    print(f"  Denoising steps: {args.denoising_steps}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Top-P: {args.top_p}")
    print(f"  Remasking strategy: {args.remasking_strategy}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    print(f"  Use KV cache: {args.use_kv_cache}")
    print(f"  Stopping criteria: {args.stopping_criteria_idx}")

    # Tokenize prompt
    origin_prompt = args.prompt

    # Tokenize without special tokens first, then add BOS manually
    tokens = tokenizer(
        origin_prompt,
        return_tensors='pt',
        add_special_tokens=False,
        max_length=args.prompt_length - 1,  # Reserve space for BOS
        truncation=True
    )

    # Manually prepend BOS token if available
    if tokenizer.bos_token_id is not None:
        bos_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long)
        tokens['input_ids'] = torch.cat([bos_id, tokens['input_ids']], dim=1)
        if 'attention_mask' in tokens:
            ones = torch.ones((1, 1), dtype=torch.long)
            tokens['attention_mask'] = torch.cat([ones, tokens['attention_mask']], dim=1)

    print(f"\nPrompt: {origin_prompt}")
    print(f"Prompt length: {tokens['input_ids'].shape[1]} tokens")
    print(f"Generation length: {args.gen_length} tokens")

    # Generate
    print("\nGenerating...")
    
    if args.use_kv_cache:
        output_ids = causal_block_diffusion_generate(
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
            use_kv_cache=True,
            verbose=args.verbose,
        )
    else:
        output_ids = causal_block_diffusion_generate_simple(
            model,
            prompt=tokens,
            mask_id=args.mask_id,
            gen_length=args.gen_length,
            block_length=args.block_length,
            denoising_steps=args.denoising_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            confidence_threshold=args.confidence_threshold,
            stopping_criteria_idx=args.stopping_criteria_idx,
            verbose=args.verbose,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    
    # Clean up mask tokens in output
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token:
        cleaned_text = output_text.replace(tokenizer.mask_token, '')
    else:
        cleaned_text = output_text.replace('<|MASK|>', '').replace('[MASK]', '')
    
    print("\n" + "=" * 60)
    print("Generated Output:")
    print("=" * 60)
    print(cleaned_text)
    print("=" * 60)
    
    if args.verbose:
        print("\nRaw output (with special tokens):")
        print(output_text)