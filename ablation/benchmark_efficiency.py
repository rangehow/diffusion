"""
Ablation 4 (Reviewer Ruxi): Controlled wall-clock efficiency comparison.

Compares CARD vs. standard AR generation (LLaMA) at matched model sizes:
- Tokens/sec vs. sequence length
- Tokens/sec vs. block_size / steps
- FLOPs estimation per generated token

Also addresses Reviewer 2Bg4's question: 
"Why does CARD have higher throughput than AR when Block=16, Steps=16?"

Usage:
    python -m ablation.benchmark_efficiency \
        --card_model_path <path> \
        --ar_model_path <path> \
        --output_dir ablation_outputs/efficiency
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from typing import Optional, Dict, List

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# CARD Generation (simplified, no KV-cache for fair comparison)
# ============================================================================

def generate_card_no_cache(
    model,
    input_ids: torch.LongTensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    steps: int,
    gumbel_temperature: float = 0.7,
    final_temperature: float = 0.2,
) -> torch.LongTensor:
    """CARD generation without KV-cache for fair FLOPs comparison."""
    batch_size, prefix_len = input_ids.shape
    device = input_ids.device

    def get_cosine_temp(step, total, base, final):
        if total <= 1: return base
        p = step / max(total - 1, 1)
        return final + (base - final) * (1 + math.cos(math.pi * p)) / 2

    def gumbel_noise(logits, temp):
        if temp <= 0: return logits
        noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        return logits / temp + noise

    # For benchmark: all tokens are real (no padding), so attention mask is all ones.
    # This is fine since we use random dummy_ids without padding.
    committed = input_ids.clone()
    committed_len = prefix_len
    generated = []
    num_generated = 0

    while num_generated < max_new_tokens:
        curr_block = min(block_size, max_new_tokens - num_generated)
        block = torch.full(
            (batch_size, curr_block), mask_token_id,
            dtype=input_ids.dtype, device=device,
        )

        effective_steps = min(steps, curr_block + 2)
        for step in range(effective_steps):
            full_seq = torch.cat([committed, block], dim=1)
            attn = torch.ones(full_seq.shape[:2], dtype=torch.long, device=device)

            outputs = model(
                input_ids=full_seq, attention_mask=attn,
                return_dict=True, causal=True,
            )

            logits = outputs.logits[:, committed_len - 1:committed_len - 1 + curr_block, :]
            temp = get_cosine_temp(step, effective_steps, gumbel_temperature, final_temperature)
            candidate = torch.argmax(gumbel_noise(logits, temp), dim=-1).to(input_ids.dtype)

            update_mask = torch.zeros(batch_size, curr_block, dtype=torch.bool, device=device)
            if step < curr_block:
                update_mask[:, step:] = True
            else:
                update_mask[:, -1:] = True

            old_block = block.clone()
            block = torch.where(update_mask, candidate, old_block)
            if step >= 1 and (block == old_block).all():
                break

        committed = torch.cat([committed, block], dim=1)
        committed_len += curr_block
        generated.append(block)
        num_generated += curr_block

    return torch.cat([input_ids] + generated, dim=1)


# ============================================================================
# FLOPs estimation
# ============================================================================

def estimate_flops_per_token_ar(num_params: int) -> float:
    """Rough FLOPs per token for AR: ~2 * num_params per generated token."""
    return 2 * num_params


def estimate_flops_per_token_card(
    num_params: int,
    block_size: int,
    steps: int,
    prefix_len: int,
    max_new_tokens: int,
) -> float:
    """
    Rough FLOPs per token for CARD.
    
    For each block, the model processes (prefix + already_generated + block_size) tokens
    for `steps` forward passes. The average context length grows as blocks are generated.
    """
    num_blocks = math.ceil(max_new_tokens / block_size)
    total_flops = 0.0
    
    for b in range(num_blocks):
        context_len = prefix_len + b * block_size + block_size
        # Each forward pass: ~2 * params * context_length (for transformer)
        flops_per_forward = 2 * num_params * context_len
        total_flops += flops_per_forward * steps
    
    flops_per_generated_token = total_flops / max_new_tokens
    return flops_per_generated_token


# ============================================================================
# Benchmark function
# ============================================================================

def benchmark_generation(
    gen_fn,
    warmup_iter: int = 3,
    test_iter: int = 10,
) -> Dict:
    """
    Run a generation function multiple times and measure throughput.
    Returns: dict with avg_time, tokens_per_sec, etc.
    """
    # Warmup
    for _ in range(warmup_iter):
        output = gen_fn()

    torch.cuda.synchronize()
    times = []
    total_tokens = 0

    for _ in range(test_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = gen_fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        # Count generated tokens
        if isinstance(output, torch.Tensor):
            total_tokens += output.numel()

    avg_time = sum(times) / len(times)
    return {
        "avg_time_sec": avg_time,
        "times": times,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Efficiency benchmark: CARD vs AR")
    parser.add_argument("--card_model_path", type=str, required=True)
    parser.add_argument("--ar_model_path", type=str, required=False, default=None,
                        help="Path to AR model (optional; skip AR benchmark if omitted)")
    parser.add_argument("--output_dir", type=str, default="ablation_outputs/efficiency")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--test_iter", type=int, default=10,
                        help="Number of timing iterations")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Test configurations ---
    # Vary sequence length
    prompt_lengths = [32, 64, 128, 256]
    max_new_tokens_list = [64, 128, 256]
    batch_sizes = [1, 8]

    # CARD-specific configs
    card_configs = [
        {"block_size": 8,  "steps": 8},
        {"block_size": 16, "steps": 8},
        {"block_size": 16, "steps": 16},
        {"block_size": 32, "steps": 8},
        {"block_size": 64, "steps": 8},
        {"block_size": 128, "steps": 8},
    ]

    # --- Load CARD model ---
    print(f"Loading CARD model from {args.card_model_path}...")
    card_tokenizer = AutoTokenizer.from_pretrained(args.card_model_path)
    if card_tokenizer.pad_token is None:
        card_tokenizer.pad_token = card_tokenizer.eos_token

    card_model = AutoModel.from_pretrained(
        args.card_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device).eval()

    mask_token_id = card_model.config.mask_token_id
    card_num_params = sum(p.numel() for p in card_model.parameters())
    print(f"  CARD params: {card_num_params:,}")

    # --- Load AR model (optional) ---
    run_ar = args.ar_model_path is not None
    ar_model = ar_tokenizer = None
    ar_num_params = 0
    if run_ar:
        print(f"Loading AR model from {args.ar_model_path}...")
        ar_tokenizer = AutoTokenizer.from_pretrained(args.ar_model_path)
        if ar_tokenizer.pad_token is None:
            ar_tokenizer.pad_token = ar_tokenizer.eos_token

        ar_model = AutoModelForCausalLM.from_pretrained(
            args.ar_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to(device).eval()

        ar_num_params = sum(p.numel() for p in ar_model.parameters())
        print(f"  AR params: {ar_num_params:,}")
    else:
        print("Skipping AR model (--ar_model_path not provided, CARD-only mode)")

    # --- Run benchmarks ---
    all_results = []

    for batch_size in batch_sizes:
        for prompt_len in prompt_lengths:
            for max_new_tokens in max_new_tokens_list:
                print(f"\n--- BS={batch_size}, Prompt={prompt_len}, Gen={max_new_tokens} ---")

                # Create dummy prompt
                dummy_ids = torch.randint(
                    100, 50000, (batch_size, prompt_len),
                    dtype=torch.long, device=device,
                )
                # Set BOS
                dummy_ids[:, 0] = card_tokenizer.bos_token_id or 1

                # ---- AR Benchmark (if enabled) ----
                if run_ar:
                    ar_ids = dummy_ids.clone()
                    ar_attn = torch.ones_like(ar_ids)

                    def ar_gen_fn():
                        with torch.inference_mode():
                            return ar_model.generate(
                                input_ids=ar_ids,
                                attention_mask=ar_attn,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=0.7,
                                pad_token_id=ar_tokenizer.pad_token_id,
                                use_cache=True,
                            )

                    ar_result = benchmark_generation(ar_gen_fn, warmup_iter=2, test_iter=args.test_iter)
                    ar_tokens_per_sec = (batch_size * max_new_tokens) / ar_result["avg_time_sec"]
                    ar_flops = estimate_flops_per_token_ar(ar_num_params)

                    print(f"  AR: {ar_tokens_per_sec:.1f} tok/s, {ar_result['avg_time_sec']:.3f}s")

                    result_entry = {
                        "batch_size": batch_size,
                        "prompt_len": prompt_len,
                        "max_new_tokens": max_new_tokens,
                        "method": "AR",
                        "block_size": 1,
                        "steps": 1,
                        "tokens_per_sec": ar_tokens_per_sec,
                        "avg_time_sec": ar_result["avg_time_sec"],
                        "est_flops_per_token": ar_flops,
                    }
                    all_results.append(result_entry)

                # ---- CARD Benchmarks ----
                for card_cfg in card_configs:
                    bs_card = card_cfg["block_size"]
                    steps = card_cfg["steps"]

                    card_ids = dummy_ids.clone()

                    def card_gen_fn(b=bs_card, s=steps):
                        with torch.inference_mode():
                            return generate_card_no_cache(
                                card_model, card_ids, mask_token_id,
                                max_new_tokens=max_new_tokens,
                                block_size=b, steps=s,
                            )

                    card_result = benchmark_generation(
                        card_gen_fn, warmup_iter=2, test_iter=args.test_iter
                    )
                    card_tok_s = (batch_size * max_new_tokens) / card_result["avg_time_sec"]
                    card_flops = estimate_flops_per_token_card(
                        card_num_params, bs_card, steps, prompt_len, max_new_tokens
                    )

                    print(f"  CARD b={bs_card} s={steps}: {card_tok_s:.1f} tok/s, "
                          f"{card_result['avg_time_sec']:.3f}s")

                    result_entry = {
                        "batch_size": batch_size,
                        "prompt_len": prompt_len,
                        "max_new_tokens": max_new_tokens,
                        "method": "CARD",
                        "block_size": bs_card,
                        "steps": steps,
                        "tokens_per_sec": card_tok_s,
                        "avg_time_sec": card_result["avg_time_sec"],
                        "est_flops_per_token": card_flops,
                    }
                    all_results.append(result_entry)

    # --- Save results ---
    output = {
        "card_model_path": args.card_model_path,
        "ar_model_path": args.ar_model_path if run_ar else None,
        "card_num_params": card_num_params,
        "ar_num_params": ar_num_params if run_ar else None,
        "timestamp": datetime.now().isoformat(),
        "test_iterations": args.test_iter,
        "results": all_results,
    }

    output_path = os.path.join(args.output_dir, "efficiency_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Print summary ---
    print(f"\n{'='*90}")
    print(f"EFFICIENCY BENCHMARK SUMMARY")
    print(f"{'='*90}")
    if run_ar:
        print(f"CARD params: {card_num_params:,}  |  AR params: {ar_num_params:,}")
    else:
        print(f"CARD params: {card_num_params:,}  (CARD-only mode)")
    print(f"{'='*90}")
    print(f"{'Method':<8} {'BS':>3} {'Block':>5} {'Steps':>5} "
          f"{'Prompt':>6} {'GenLen':>6} {'Tok/s':>10} {'Time(s)':>8} {'FLOPs/tok':>14}")
    print(f"{'-'*80}")

    for r in all_results:
        print(f"{r['method']:<8} {r['batch_size']:>3} {r['block_size']:>5} "
              f"{r['steps']:>5} {r['prompt_len']:>6} {r['max_new_tokens']:>6} "
              f"{r['tokens_per_sec']:>10.1f} {r['avg_time_sec']:>8.3f} "
              f"{r['est_flops_per_token']:>14.2e}")

    print(f"{'='*90}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
