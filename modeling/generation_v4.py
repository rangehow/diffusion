"""
V4 Inference with Re-Reordering: Every denoising step matches V4 training distribution.

Core insight: V4 trains on [observed → masked] layout. During multi-step inference,
after filling some positions, the block becomes interleaved (observed & masked mixed).
Re-reordering the block before each forward pass restores the training-consistent layout.

This is purely an inference-time change — no training modification needed.
"""

import torch
from typing import Optional


def generate_v4_reorder(
    model,
    input_ids: torch.LongTensor,
    mask_token_id: int,
    max_new_tokens: int = 64,
    block_size: int = 16,
    num_diffusion_steps: int = 8,
    temperature: float = 1.0,
    confidence_threshold: float = 0.5,
    do_sample: bool = False,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    decode_top_k_positions: Optional[int] = None,
) -> torch.LongTensor:
    """
    Block-wise generation with V4-style re-reordering at every denoising step.
    
    At each denoising step:
      1. Partition block into [observed, masked] (preserving relative order within each group)
      2. Prepend to clean prefix, use sequential position IDs (model default)
      3. Forward pass with standard causal attention
      4. Map predictions back to logical positions
      5. Fill confident predictions
    
    This ensures EVERY forward pass sees the exact pattern the V4 collator produces:
    [clean prefix → observed block tokens → masked block tokens → EOS]
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device
    assert batch_size == 1, "V4 reorder generation currently supports batch_size=1"
    
    prefix = input_ids[0].tolist()  # Clean prompt tokens
    generated = []                   # Will accumulate generated tokens
    
    num_generated = 0
    while num_generated < max_new_tokens:
        current_block_size = min(block_size, max_new_tokens - num_generated)
        if current_block_size <= 0:
            break
        
        # Initialize block: all masks, with logical positions
        # block[i] = (logical_position, token_value)
        block_start_pos = len(prefix) + num_generated
        block = [mask_token_id] * current_block_size
        block_logical_pos = list(range(block_start_pos, block_start_pos + current_block_size))
        
        for step in range(num_diffusion_steps):
            # ──── Step A: Partition block into observed and masked ────
            observed_items = []   # (logical_pos, token_value)
            masked_items = []     # (logical_pos,)
            for lpos, tok in zip(block_logical_pos, block):
                if tok != mask_token_id:
                    observed_items.append((lpos, tok))
                else:
                    masked_items.append(lpos)
            
            if not masked_items:
                break  # All positions filled
            
            # ──── Step B: Build re-reordered input sequence ────
            # Layout: [prefix + previously_generated | observed_block | masked_block]
            # All with sequential position IDs (model default)
            
            full_prefix = prefix + generated  # Clean prefix + all previously committed tokens
            reordered_block = (
                [tok for _, tok in observed_items] +    # Observed block tokens
                [mask_token_id] * len(masked_items)     # Remaining masks
            )
            
            input_seq = full_prefix + reordered_block
            input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_tensor, dtype=torch.long)
            
            # ──── Step C: Forward pass (standard causal, sequential positions) ────
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    return_dict=True,
                    causal=True,
                    use_daum=False,
                )
            logits = outputs.logits  # (1, seq_len, vocab)
            
            # ──── Step D: Extract predictions for mask positions ────
            # Due to shifted causal: output at position i predicts token at position i+1
            # The first mask (at physical position len(prefix)+len(observed)) 
            # is predicted by position len(prefix)+len(observed)-1
            
            mask_phys_start = len(full_prefix) + len(observed_items)
            num_masks = len(masked_items)
            
            # Predictions for masks:
            # mask[0] ← logits[mask_phys_start - 1]
            # mask[k] ← logits[mask_phys_start - 1 + k]
            pred_start = mask_phys_start - 1
            mask_logits = logits[0, pred_start : pred_start + num_masks, :]  # (num_masks, vocab)
            
            # ──── Step E: Sample/argmax and compute confidence ────
            if do_sample:
                scaled_logits = mask_logits / temperature
                if top_k is not None:
                    topk_vals, topk_idx = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits = torch.full_like(scaled_logits, float('-inf'))
                    scaled_logits.scatter_(-1, topk_idx, topk_vals)
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(scaled_logits, descending=True, dim=-1)
                    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs > top_p
                    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                    remove_mask[..., 0] = 0
                    remove_idx = remove_mask.scatter(-1, sorted_idx, remove_mask)
                    scaled_logits = scaled_logits.masked_fill(remove_idx, float('-inf'))
                probs = torch.softmax(scaled_logits, dim=-1)
                pred_tokens = torch.multinomial(probs, 1).squeeze(-1)
            else:
                probs = torch.softmax(mask_logits, dim=-1)
                pred_tokens = torch.argmax(mask_logits, dim=-1)
            
            confidence, _ = probs.max(dim=-1)
            
            # ──── Step F: Fill confident predictions back into block ────
            # Map physical mask index k → logical position masked_items[k]
            
            if decode_top_k_positions is not None:
                # Top-k most confident masks
                k = min(decode_top_k_positions, num_masks)
                _, topk_mask_indices = torch.topk(confidence, k)
                fill_indices = topk_mask_indices.tolist()
            else:
                # Fill all (or use confidence threshold)
                fill_indices = [k for k in range(num_masks) 
                               if confidence[k].item() > confidence_threshold or step == num_diffusion_steps - 1]
            
            for k in fill_indices:
                logical_pos = masked_items[k]
                block_idx = block_logical_pos.index(logical_pos)
                block[block_idx] = pred_tokens[k].item()
            
            # Force-fill the first remaining mask (AR progression guarantee)
            remaining_masks_after = [i for i, tok in enumerate(block) if tok == mask_token_id]
            if remaining_masks_after and 0 not in fill_indices:
                # Find the first mask in logical order and fill it
                first_mask_block_idx = remaining_masks_after[0]
                # Find which physical mask index this corresponds to
                first_mask_lpos = block_logical_pos[first_mask_block_idx]
                if first_mask_lpos in masked_items:
                    k = masked_items.index(first_mask_lpos)
                    block[first_mask_block_idx] = pred_tokens[k].item()
            
            # Check if block is complete
            if all(tok != mask_token_id for tok in block):
                break
        
        # ──── Force-fill any remaining masks in last step ────
        if any(tok == mask_token_id for tok in block):
            # Run one final pass to fill everything
            full_prefix_final = prefix + generated
            observed_final = [(lp, t) for lp, t in zip(block_logical_pos, block) if t != mask_token_id]
            masked_final = [lp for lp, t in zip(block_logical_pos, block) if t == mask_token_id]
            
            reordered = [t for _, t in observed_final] + [mask_token_id] * len(masked_final)
            input_seq = full_prefix_final + reordered
            input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
            
            with torch.inference_mode():
                outputs = model(input_ids=input_tensor, attention_mask=torch.ones_like(input_tensor),
                              return_dict=True, causal=True, use_daum=False)
            logits = outputs.logits
            mask_start = len(full_prefix_final) + len(observed_final)
            for k, lpos in enumerate(masked_final):
                pred_tok = torch.argmax(logits[0, mask_start - 1 + k]).item()
                block_idx = block_logical_pos.index(lpos)
                block[block_idx] = pred_tok
        
        # ──── Commit block to generated sequence ────
        generated.extend(block)
        num_generated += current_block_size
    
    # Build final output
    full_output = prefix + generated
    return torch.tensor([full_output], dtype=torch.long, device=device)
