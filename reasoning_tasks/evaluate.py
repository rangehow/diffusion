"""
Evaluation script for reasoning tasks (Countdown, Sudoku).

Loads a trained model checkpoint, generates responses for each test example,
and computes exact-match accuracy.

Supports three model types:
- CARD (ModernBERT with shifted causal diffusion): iterative block denoising
- CARD_AR: CARD model generating token-by-token (block_size=1, like AR)
- AR (LlamaForCausalLM): standard autoregressive generation
- MDLM (LLaDA): iterative unmasking

All modes use BATCHED inference for GPU utilization.
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
import datasets


# ============================================================================
# Generation Utilities
# ============================================================================

def top_k_top_p_filter(logits, top_k=50, top_p=0.95):
    """Filter logits with top-k and top-p (nucleus)."""
    if top_k > 0:
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[..., -1:]
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        indices_to_remove = torch.scatter(
            torch.zeros_like(logits, dtype=torch.bool), -1, sorted_indices, mask
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits


@torch.no_grad()
def generate_ar_batch(model, tokenizer, batch_prompt_ids, max_new_tokens=128,
                      temperature=0.0, top_k=0, top_p=1.0):
    """
    Batched autoregressive generation.
    batch_prompt_ids: list of 1D tensors (variable length prompts)
    Returns: list of 1D tensors (generated tokens only, excluding prompt)
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # Pad prompts to same length (left-pad for AR)
    max_prompt_len = max(p.shape[0] for p in batch_prompt_ids)
    batch_size = len(batch_prompt_ids)
    prompt_lens = [p.shape[0] for p in batch_prompt_ids]
    
    # Left-pad: put padding on the left so generation continues from the right
    input_ids = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=device)
    for i, (p, plen) in enumerate(zip(batch_prompt_ids, prompt_lens)):
        input_ids[i, max_prompt_len - plen:] = p.to(device)
        attention_mask[i, max_prompt_len - plen:] = 1
    
    # Track which sequences are still generating
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    generated = [[] for _ in range(batch_size)]
    
    for step in range(max_new_tokens):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        next_logits = outputs.logits[:, -1, :]  # [B, V]
        
        if temperature == 0.0:
            next_tokens = next_logits.argmax(dim=-1)  # [B]
        else:
            next_logits = next_logits / temperature
            next_logits = top_k_top_p_filter(next_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Append tokens
        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=torch.long, device=device)], dim=-1)
        
        for i in range(batch_size):
            if active[i]:
                tok = next_tokens[i].item()
                generated[i].append(tok)
                if tok == tokenizer.eos_token_id:
                    active[i] = False
        
        if not active.any():
            break
    
    return [torch.tensor(g, dtype=torch.long) for g in generated]


@torch.no_grad()
def generate_card_batch(model, tokenizer, batch_prompt_ids, max_new_tokens=128,
                        block_size=32, num_steps=16, temperature=0.0):
    """
    Batched CARD generation: append [MASK] tokens and iteratively denoise.
    Uses confidence-based decoding, processes all examples in parallel.
    
    When block_size=1, this becomes token-by-token generation (CARD as AR).
    """
    device = next(model.parameters()).device
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_size = len(batch_prompt_ids)
    prompt_lens = [p.shape[0] for p in batch_prompt_ids]
    
    all_generated = [[] for _ in range(batch_size)]
    
    # For token-by-token (block_size=1), we loop max_new_tokens times
    # For block generation, we loop ceil(max_new_tokens / block_size) times
    actual_block_size = max(1, block_size)
    n_blocks = (max_new_tokens + actual_block_size - 1) // actual_block_size
    
    # Build initial sequences (just prompts)
    current_seqs = [p.clone() for p in batch_prompt_ids]
    
    for block_idx in range(n_blocks):
        remaining = max_new_tokens - len(all_generated[0]) if all_generated[0] else max_new_tokens
        if remaining <= 0:
            break
        this_block = min(actual_block_size, remaining)
        
        # Pad all current sequences + mask block to same length
        max_seq_len = max(s.shape[0] for s in current_seqs) + this_block
        input_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=device)
        
        block_start_positions = []
        for i, seq in enumerate(current_seqs):
            slen = seq.shape[0]
            input_ids[i, :slen] = seq.to(device)
            input_ids[i, slen:slen + this_block] = mask_id
            attention_mask[i, :slen + this_block] = 1
            block_start_positions.append(slen)
        
        # Iterative denoising
        actual_steps = num_steps if this_block > 1 else 1  # 1 step for token-by-token
        for step in range(actual_steps):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, seq_len, V]
            
            # Prevent predicting [MASK] token
            logits[..., mask_id] = float('-inf')
            
            # CARD shifted causal: prediction for pos comes from logits[pos-1]
            # Process all examples in batch at once (vectorized)
            for i in range(batch_size):
                bstart = block_start_positions[i]
                block_region = input_ids[i, bstart:bstart + this_block]
                mask_flags = (block_region == mask_id)
                mask_positions = mask_flags.nonzero(as_tuple=True)[0]  # relative to bstart
                
                if len(mask_positions) == 0:
                    continue
                
                # Get logits for prediction positions (shifted by -1)
                abs_mask_pos = mask_positions + bstart
                pred_pos = torch.clamp(abs_mask_pos - 1, min=0)
                pred_logits_all = logits[i, pred_pos]  # [n_masks, V]
                
                if step == actual_steps - 1:
                    # Last step: force decode all remaining masks
                    if temperature == 0.0:
                        tokens = pred_logits_all.argmax(dim=-1)
                    else:
                        probs = F.softmax(pred_logits_all / temperature, dim=-1)
                        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    input_ids[i, abs_mask_pos] = tokens
                else:
                    n_to_decode = max(1, len(mask_positions) // (actual_steps - step))
                    # Compute confidence for all mask positions at once
                    probs_all = F.softmax(pred_logits_all, dim=-1)
                    max_probs = probs_all.max(dim=-1).values  # [n_masks]
                    # Sort by confidence (highest first)
                    _, sorted_idx = max_probs.sort(descending=True)
                    top_idx = sorted_idx[:n_to_decode]
                    top_logits = pred_logits_all[top_idx]
                    top_abs_pos = abs_mask_pos[top_idx]
                    
                    if temperature == 0.0:
                        tokens = top_logits.argmax(dim=-1)
                    else:
                        probs = F.softmax(top_logits / temperature, dim=-1)
                        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    input_ids[i, top_abs_pos] = tokens
        
        # Collect generated block tokens and extend current sequences
        for i in range(batch_size):
            bstart = block_start_positions[i]
            block_tokens = input_ids[i, bstart:bstart + this_block].cpu().tolist()
            all_generated[i].extend(block_tokens)
            # Update current sequence for next block
            current_seqs[i] = input_ids[i, :bstart + this_block].cpu()
    
    # Post-process: remove trailing masks
    results = []
    for gen in all_generated:
        valid = []
        for tok in gen:
            if tok == mask_id:
                continue
            if tok == tokenizer.eos_token_id:
                break
            valid.append(tok)
        results.append(torch.tensor(valid, dtype=torch.long))
    return results


@torch.no_grad()
def generate_mdlm_batch(model, tokenizer, batch_prompt_ids, max_new_tokens=128,
                         num_steps=64, temperature=0.0, max_seq_len=None,
                         decoding_strategy='deterministic-linear'):
    """
    Batched MDLM/MDM generation using TopK decoding.
    
    Supports two strategies (matching Ye et al. 2024):
    - 'deterministic-linear': deterministic top-k, linear remask schedule
    - 'stochastic0.5-linear': stochastic top-k with Gumbel noise (noise_scale=0.5)
    
    The generation process:
    1. Start with all response positions as [MASK]  
    2. Each step t (from T-1 down to 0):
       - Forward pass → get predictions and confidence scores
       - Fill ALL masked positions with predictions (x0)
       - If t > 0: re-mask the least confident positions according to schedule
    3. Linear schedule: re-mask rate = t/T (decreases from ~1 to 0)
    
    max_seq_len: total sequence length matching training's max_length.
    """
    device = next(model.parameters()).device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch_size = len(batch_prompt_ids)
    prompt_lens = [p.shape[0] for p in batch_prompt_ids]
    
    max_prompt_len = max(prompt_lens)
    
    # Parse decoding strategy
    parts = decoding_strategy.split('-')
    topk_mode = parts[0]  # 'deterministic' or 'stochastic0.5'
    schedule = parts[1] if len(parts) > 1 else 'linear'
    stochastic = topk_mode.startswith('stochastic')
    noise_scale = float(topk_mode.replace('stochastic', '')) if stochastic else 0.0
    
    if max_seq_len is not None:
        total_len = max_seq_len
    else:
        gen_len_default = max_new_tokens + 1
        total_len = max_prompt_len + gen_len_default
    
    input_ids = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long, device=device)
    # Track which positions are maskable (the generation region)
    init_maskable = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
    
    for i, (p, plen) in enumerate(zip(batch_prompt_ids, prompt_lens)):
        input_ids[i, :plen] = p.to(device)
        gen_end = min(plen + max_new_tokens + 1, total_len)
        input_ids[i, plen:gen_end] = mask_id
        attention_mask[i, :gen_end] = 1
        init_maskable[i, plen:gen_end] = True
    
    # Generate: iterate from t=T-1 down to t=0 (matching Ye et al.)
    for t in range(num_steps - 1, -1, -1):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, total_len, V]
        
        # Prevent predicting [MASK] and [PAD]
        logits[..., mask_id] = float('-inf')
        logits[..., pad_id] = float('-inf')
        
        # Compute log-softmax scores for confidence
        log_scores = torch.log_softmax(logits, dim=-1)  # [B, total_len, V]
        
        # Get best prediction and its confidence at every position
        x0_scores, x0_tokens = log_scores.max(dim=-1)  # [B, total_len]
        
        # Keep non-[MASK] positions unchanged, update [MASK] positions
        is_masked = (input_ids == mask_id) & init_maskable
        x0 = input_ids.clone()
        x0[is_masked] = x0_tokens[is_masked]
        
        if t > 0:
            # Re-mask based on schedule: rate determines fraction to re-mask
            if schedule == 'linear':
                rate = t / num_steps
            elif schedule == 'cosine':
                import numpy as np
                rate = np.cos((num_steps - t) / num_steps * np.pi * 0.5)
            else:
                rate = t / num_steps
            
            for i in range(batch_size):
                maskable_pos = init_maskable[i].nonzero(as_tuple=True)[0]
                if len(maskable_pos) == 0:
                    continue
                
                # Compute cutoff: re-mask this many positions
                n_remask = int(len(maskable_pos) * rate)
                if n_remask == 0:
                    continue
                
                # Get scores at maskable positions (not just currently-masked)
                pos_scores = x0_scores[i, maskable_pos]
                
                # Set unmaskable positions to high score so they're never selected
                # (they're already handled by init_maskable filtering)
                
                if stochastic:
                    # Add Gumbel noise to scores for stochastic selection
                    gumbel_noise = -torch.log(-torch.log(torch.rand_like(pos_scores) + 1e-8) + 1e-8)
                    noisy_scores = pos_scores + noise_scale * rate * gumbel_noise
                else:
                    noisy_scores = pos_scores
                
                # Select the n_remask positions with LOWEST scores to re-mask
                _, sorted_idx = noisy_scores.sort()
                remask_local_idx = sorted_idx[:n_remask]
                remask_abs_idx = maskable_pos[remask_local_idx]
                x0[i, remask_abs_idx] = mask_id
        
        input_ids = x0
    
    # Extract generated tokens per example
    results = []
    for i in range(batch_size):
        plen = prompt_lens[i]
        gen_end = min(plen + max_new_tokens + 1, total_len)
        gen_tokens = input_ids[i, plen:gen_end].cpu().tolist()
        valid = []
        for tok in gen_tokens:
            if tok == mask_id:
                continue
            if eos_id is not None and tok == eos_id:
                break
            if tok == pad_id:
                break
            valid.append(tok)
        results.append(torch.tensor(valid, dtype=torch.long))
    return results


# ============================================================================
# Evaluation Logic
# ============================================================================

SEP_TOKEN_TEXT = " [SEP] "


def normalize_answer(text: str) -> str:
    """Normalize answer string for comparison."""
    return text.strip().lower()


def check_countdown_answer(prediction: str, target_response: str, prompt: str = "") -> bool:
    """
    Check if a Countdown prediction is correct.
    A prediction is correct if:
    1. It exactly matches the gold answer, OR
    2. Every arithmetic step is valid AND the final result equals the target number.
    """
    pred = normalize_answer(prediction)
    gold = normalize_answer(target_response)
    
    # Exact match
    if pred == gold:
        return True
    
    # Empty prediction for trivial cases (target is one of the numbers)
    if pred == "" and gold == "":
        return True
    
    # Parse target number from prompt
    # Ye et al. format: "86,28,13,31,96" — target is the LAST number
    # Old format: "Numbers: 6 24 10 20, Target: 46"
    target_num = None
    try:
        if 'target:' in prompt.lower():
            target_num = int(prompt.lower().split('target:')[1].strip())
        else:
            # Ye et al. compact format: last comma-separated value is target
            parts = prompt.strip().split(',')
            if parts:
                target_num = int(parts[-1].strip())
    except:
        pass
    
    # Fall back to gold's final result
    if target_num is None:
        try:
            gold_steps = [s.strip() for s in gold.split(',') if s.strip()]
            if gold_steps:
                target_num = int(gold_steps[-1].split('=')[-1].strip())
        except:
            pass
    
    if target_num is None:
        return False
    
    # Verify prediction: each step must be arithmetically correct,
    # and final result must equal target
    try:
        pred_steps = [s.strip() for s in pred.split(',') if s.strip()]
        if not pred_steps:
            return False
        
        for step in pred_steps:
            if '=' not in step:
                return False
            expr, result_str = step.rsplit('=', 1)
            result = int(result_str.strip())
            computed = eval(expr.strip())
            if computed != result:
                return False
        
        # Check final result
        final_result = int(pred_steps[-1].rsplit('=', 1)[-1].strip())
        return final_result == target_num
    except Exception:
        pass
    
    return False


def check_sudoku_answer(prediction: str, target_response: str) -> dict:
    """
    Check Sudoku prediction against target.
    Returns dict with:
      - 'board_correct': bool (exact match on all cells)
      - 'cell_accuracy': float (fraction of correctly predicted cells)
    """
    pred = ''.join(c for c in prediction if c.isdigit())
    gold = ''.join(c for c in target_response if c.isdigit())
    
    board_correct = (pred == gold)
    
    # Cell accuracy: fraction of matching digits
    if len(gold) == 0:
        cell_accuracy = 1.0 if len(pred) == 0 else 0.0
    else:
        n_match = sum(1 for a, b in zip(pred, gold) if a == b)
        # Penalize length mismatch
        cell_accuracy = n_match / len(gold) if len(pred) >= len(gold) else n_match / len(gold)
    
    return {'board_correct': board_correct, 'cell_accuracy': cell_accuracy}


def evaluate_model(model, tokenizer, test_dataset, mode, task_type,
                   max_new_tokens=128, num_steps=16, block_size=32,
                   temperature=0.0, max_examples=None, batch_size=64,
                   max_seq_len=None, decoding_strategy='deterministic-linear'):
    """
    Evaluate a model on a reasoning task test set with BATCHED inference.
    Returns accuracy and per-example results.
    """
    device = next(model.parameters()).device
    model.eval()
    
    correct = 0
    total = 0
    results = []
    
    n_examples = len(test_dataset)
    if max_examples:
        n_examples = min(n_examples, max_examples)
    
    # Prepare all prompts
    all_prompts = []
    all_targets = []
    all_prompt_ids = []
    
    print(f"[INFO] Tokenizing {n_examples} examples ...")
    t0 = time.time()
    for i in range(n_examples):
        example = test_dataset[i]
        prompt_text = example["prompt"]
        target_response = example["response"]
        
        # During training, the collator splits text on " [SEP] " and tokenizes
        # prompt and response SEPARATELY — the [SEP] token itself is NEVER
        # part of the token sequence. So the eval prompt must NOT include [SEP].
        # Training format: [BOS] prompt_tokens response_tokens [EOS]
        if tokenizer.bos_token:
            prompt_text_enc = tokenizer.bos_token + prompt_text
        else:
            prompt_text_enc = prompt_text
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt_text_enc, add_special_tokens=False),
            dtype=torch.long
        )
        all_prompts.append(prompt_text)
        all_targets.append(target_response)
        all_prompt_ids.append(prompt_ids)
    
    print(f"[INFO] Tokenization done in {time.time()-t0:.1f}s")
    print(f"[INFO] Prompt lengths: min={min(p.shape[0] for p in all_prompt_ids)}, "
          f"max={max(p.shape[0] for p in all_prompt_ids)}, "
          f"mean={sum(p.shape[0] for p in all_prompt_ids)/len(all_prompt_ids):.1f}")
    
    # Process in batches
    n_batches = (n_examples + batch_size - 1) // batch_size
    print(f"[INFO] Running inference: {n_examples} examples, batch_size={batch_size}, {n_batches} batches")
    print(f"[INFO] Mode: {mode}, max_new_tokens={max_new_tokens}, "
          f"num_steps={num_steps}, block_size={block_size}, temperature={temperature}")
    
    total_gen_time = 0.0
    total_tokens_generated = 0
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_examples)
        batch_prompt_ids = all_prompt_ids[batch_start:batch_end]
        actual_bs = len(batch_prompt_ids)
        
        t_batch_start = time.time()
        
        # Generate
        mode_base = mode.replace("reasoning_", "")
        if mode_base == "ar":
            batch_generated = generate_ar_batch(
                model, tokenizer, batch_prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        elif mode_base == "card":
            batch_generated = generate_card_batch(
                model, tokenizer, batch_prompt_ids,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                num_steps=num_steps,
                temperature=temperature
            )
        elif mode_base == "card_ar":
            # CARD model, but generate token-by-token (block_size=1)
            batch_generated = generate_card_batch(
                model, tokenizer, batch_prompt_ids,
                max_new_tokens=max_new_tokens,
                block_size=1,
                num_steps=1,
                temperature=temperature
            )
        elif mode_base in ("mdlm", "mdm"):
            batch_generated = generate_mdlm_batch(
                model, tokenizer, batch_prompt_ids,
                max_new_tokens=max_new_tokens,
                num_steps=num_steps,
                temperature=temperature,
                max_seq_len=max_seq_len,
                decoding_strategy=decoding_strategy
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        t_batch_end = time.time()
        batch_time = t_batch_end - t_batch_start
        total_gen_time += batch_time
        
        # Evaluate batch
        batch_correct = 0
        for j, gen_ids in enumerate(batch_generated):
            idx = batch_start + j
            prediction = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            target_response = all_targets[idx]
            
            total_tokens_generated += len(gen_ids)
            
            if task_type.startswith("cd"):
                is_correct = check_countdown_answer(prediction, target_response, prompt=all_prompts[idx])
                cell_acc = None
            elif task_type == "sudoku":
                sudoku_result = check_sudoku_answer(prediction, target_response)
                is_correct = sudoku_result['board_correct']
                cell_acc = sudoku_result['cell_accuracy']
            else:
                is_correct = normalize_answer(prediction) == normalize_answer(target_response)
                cell_acc = None
            
            if is_correct:
                correct += 1
                batch_correct += 1
            total += 1
            
            result_entry = {
                "prompt": all_prompts[idx],
                "target": target_response,
                "prediction": prediction,
                "correct": is_correct,
            }
            if cell_acc is not None:
                result_entry["cell_accuracy"] = cell_acc
            results.append(result_entry)
        
        # Batch logging
        batch_acc = batch_correct / actual_bs
        running_acc = correct / total
        tok_per_sec = total_tokens_generated / total_gen_time if total_gen_time > 0 else 0
        print(f"[BATCH {batch_idx+1}/{n_batches}] "
              f"size={actual_bs}, time={batch_time:.1f}s, "
              f"batch_acc={batch_acc:.1%}, running_acc={running_acc:.1%} ({correct}/{total}), "
              f"throughput={tok_per_sec:.0f} tok/s")
        
        # Show first few examples from first batch
        if batch_idx == 0:
            n_show = min(3, actual_bs)
            print(f"  --- First {n_show} predictions ---")
            for j in range(n_show):
                r = results[j]
                status = "✓" if r["correct"] else "✗"
                print(f"  [{status}] Target:     {r['target'][:80]}")
                print(f"       Prediction: {r['prediction'][:80]}")
    
    accuracy = correct / total if total > 0 else 0.0
    throughput = total_tokens_generated / total_gen_time if total_gen_time > 0 else 0.0
    
    # Compute cell accuracy for Sudoku
    cell_accs = [r['cell_accuracy'] for r in results if 'cell_accuracy' in r]
    avg_cell_acc = sum(cell_accs) / len(cell_accs) if cell_accs else None
    
    print(f"\n[SUMMARY] Total time: {total_gen_time:.1f}s, "
          f"Total tokens: {total_tokens_generated}, "
          f"Throughput: {throughput:.0f} tok/s")
    if avg_cell_acc is not None:
        print(f"[SUMMARY] Sudoku cell accuracy: {avg_cell_acc:.1%}")
    
    return accuracy, results, total_gen_time, total_tokens_generated, throughput, avg_cell_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning tasks")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--task", type=str, required=True,
                        choices=["cd3", "cd4", "cd5", "sudoku"])
    parser.add_argument("--mode", type=str, required=True,
                        choices=["ar", "reasoning_ar", "card", "reasoning_card",
                                 "card_ar", "reasoning_card_ar",
                                 "mdlm", "reasoning_mdlm",
                                 "mdm", "reasoning_mdm"])
    parser.add_argument("--data_dir", type=str,
                        default="reasoning_tasks/data")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save detailed results JSON")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="Total sequence length matching training max_length. "
                             "For MDLM: pad to this length with PAD/attn=0 to match training distribution.")
    parser.add_argument("--num_steps", type=int, default=64,
                        help="Diffusion denoising steps (for CARD/MDLM)")
    parser.add_argument("--block_size", type=int, default=64,
                        help="Block size for CARD generation")
    parser.add_argument("--decoding_strategy", type=str, default="deterministic-linear",
                        help="Decoding strategy for MDLM/MDM. "
                             "'deterministic-linear' or 'stochastic0.5-linear' (Ye et al.)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Max test examples to evaluate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference (default 64)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Load tokenizer — always use our custom char tokenizer for reasoning tasks
    from reasoning_tasks.char_tokenizer import ReasoningCharTokenizer
    tokenizer = ReasoningCharTokenizer()
    print(f"[INFO] Using ReasoningCharTokenizer (vocab_size={tokenizer.vocab_size})")
    
    # Load model
    mode_base = args.mode.replace("reasoning_", "")
    if mode_base == "ar":
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    elif mode_base in ("mdlm", "mdm"):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from llada.modeling_llada import LLaDAModelLM
        from llada.configuration_llada import LLaDAConfig
        config = LLaDAConfig.from_pretrained(args.model_path)
        model = LLaDAModelLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.bfloat16)
    elif mode_base in ("card", "card_ar"):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from modeling.modeling_niu import ModernBertForDiffusionLM
        from modeling.configuration_niu import NiuConfig
        config = NiuConfig.from_pretrained(args.model_path)
        model = ModernBertForDiffusionLM.from_pretrained(args.model_path, config=config, torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unknown mode base: {mode_base}")
    
    model = model.to(args.device)
    model.eval()
    
    # Load test dataset
    test_path = os.path.join(args.data_dir, f"{args.task}_test")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found: {test_path}")
    test_dataset = datasets.load_from_disk(test_path)
    
    print(f"\n{'='*70}")
    print(f"  Evaluating {args.mode} on {args.task}")
    print(f"  Model: {args.model_path}")
    print(f"  Test examples: {len(test_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    if mode_base in ("card", "card_ar", "mdlm", "mdm"):
        print(f"  Diffusion steps: {args.num_steps}")
    if mode_base == "card":
        print(f"  Block size: {args.block_size}")
    if mode_base == "card_ar":
        print(f"  Block size: 1 (token-by-token, CARD as AR)")
    print(f"  Temperature: {args.temperature}")
    print(f"  Device: {args.device}")
    print(f"{'='*70}\n")
    
    t_start = time.time()
    
    # Evaluate
    accuracy, results, gen_time, gen_tokens, throughput, cell_accuracy = evaluate_model(
        model, tokenizer, test_dataset,
        mode=args.mode,
        task_type=args.task,
        max_new_tokens=args.max_new_tokens,
        num_steps=args.num_steps,
        block_size=args.block_size,
        temperature=args.temperature,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        decoding_strategy=args.decoding_strategy,
    )
    
    t_total = time.time() - t_start
    
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS: {args.task} ({args.mode})")
    print(f"  Accuracy: {accuracy:.4f} ({int(accuracy * len(results))}/{len(results)})")
    if cell_accuracy is not None:
        print(f"  Cell Accuracy: {cell_accuracy:.4f}")
    print(f"  Wall time: {t_total:.1f}s")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Tokens generated: {gen_tokens}")
    print(f"  Throughput: {throughput:.0f} tok/s")
    print(f"{'='*70}")
    
    # Show some examples
    print("\nSample predictions (first 5):")
    for i, r in enumerate(results[:5]):
        status = "✓" if r["correct"] else "✗"
        print(f"  [{status}] Prompt:     {r['prompt'][:70]}")
        print(f"       Target:     {r['target'][:70]}")
        print(f"       Prediction: {r['prediction'][:70]}")
    
    # Save results
    if args.output_file:
        output = {
            "task": args.task,
            "mode": args.mode,
            "model_path": args.model_path,
            "accuracy": accuracy,
            "n_correct": int(accuracy * len(results)),
            "n_total": len(results),
            "wall_time_seconds": t_total,
            "gen_time_seconds": gen_time,
            "tokens_generated": gen_tokens,
            "throughput_tok_per_sec": throughput,
            "cell_accuracy": cell_accuracy,
            "config": {
                "max_new_tokens": args.max_new_tokens,
                "num_steps": args.num_steps,
                "block_size": args.block_size,
                "temperature": args.temperature,
                "batch_size": args.batch_size,
            },
            "results": results,
        }
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output_file}")
    
    return accuracy


if __name__ == "__main__":
    main()
