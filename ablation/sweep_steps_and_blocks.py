"""
Ablation 2 & 3: Sweep diffusion steps and block sizes at inference time.

- Ablation 2 (Reviewer YsYm): Sensitivity to number of diffusion steps
- Ablation 3 (Reviewer mkMw): Block size vs. quality trade-off

For each (block_size, num_steps) pair, generates text from HellaSwag prompts 
and measures:
  - Generation quality (PPL via external LM)
  - Throughput (tokens/sec)
  - Repetition rate

Usage:
    python -m ablation.sweep_steps_and_blocks \
        --model_path <CARD_checkpoint_path> \
        --output_dir ablation_outputs/sweep_results \
        [--block_sizes 4,8,16,32,64,128] \
        [--step_counts 1,2,4,8,16,32] \
        [--max_new_tokens 128] \
        [--num_samples 500]
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from typing import List, Dict, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm


# ============================================================================
# Generation function (cosine schedule, consistent with existing codebase)
# ============================================================================

def generate_diffusion_quality(
    model,
    input_ids: torch.LongTensor,
    mask_token_id: int,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 128,
    steps: int = 8,
    block_size: int = 16,
    gumbel_temperature: float = 0.7,
    final_temperature: float = 0.2,
    min_refinement_steps: int = 2,
) -> torch.LongTensor:
    """Quality-optimized generation with cosine temperature annealing."""
    batch_size, prefix_len = input_ids.shape
    device, dtype = input_ids.device, model.lm_head.weight.dtype

    def get_cosine_temperature(step, total_steps, base_temp, final_temp):
        if total_steps <= 1:
            return base_temp
        progress = step / max(total_steps - 1, 1)
        return final_temp + (base_temp - final_temp) * (1 + math.cos(math.pi * progress)) / 2

    def add_gumbel_noise(logits, temp):
        if temp <= 0:
            return logits
        noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        return logits / temp + noise

    # Track real attention mask (important for batches with left-padding)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    committed_attn = attention_mask.clone()

    committed = input_ids.clone()
    committed_len = prefix_len
    generated = []
    num_generated = 0

    while num_generated < max_new_tokens:
        curr_block = min(block_size, max_new_tokens - num_generated)
        block = torch.full(
            (batch_size, curr_block), mask_token_id,
            dtype=input_ids.dtype, device=device
        )

        # New tokens are always real (attention=1)
        block_attn = torch.ones(batch_size, curr_block, dtype=torch.long, device=device)

        last_block_state = None
        effective_steps = min(steps, curr_block + min_refinement_steps)

        for step in range(effective_steps):
            ar_pointer = step
            if ar_pointer >= curr_block and step >= min_refinement_steps:
                if last_block_state is not None and (block == last_block_state).all():
                    break

            current_temp = get_cosine_temperature(
                step, effective_steps, gumbel_temperature, final_temperature
            )

            block_input = torch.cat([committed, block], dim=1)
            attn_mask = torch.cat([committed_attn, block_attn], dim=1)

            with torch.inference_mode():
                outputs = model(
                    input_ids=block_input,
                    attention_mask=attn_mask,
                    return_dict=True,
                    causal=True,
                )

            logits = outputs.logits[:, committed_len - 1:committed_len - 1 + curr_block, :]
            gumbel_logits = add_gumbel_noise(logits, current_temp)
            candidate_tokens = torch.argmax(gumbel_logits, dim=-1).to(input_ids.dtype)

            old_block = block.clone()
            update_mask = torch.zeros(batch_size, curr_block, dtype=torch.bool, device=device)
            if ar_pointer < curr_block:
                update_mask[:, ar_pointer:] = True
            else:
                update_mask[:, -1:] = True

            block = torch.where(update_mask, candidate_tokens, old_block)
            last_block_state = old_block

            if step >= min_refinement_steps - 1 and (block == old_block).all():
                break

        committed_len += curr_block
        committed = torch.cat([committed, block], dim=1)
        committed_attn = torch.cat([committed_attn, block_attn], dim=1)
        generated.append(block)
        num_generated += curr_block

    return torch.cat([input_ids] + generated, dim=1)


# ============================================================================
# Metrics
# ============================================================================

def compute_repetition_rate(text: str, n: int = 4) -> float:
    """Compute n-gram repetition rate."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique = set(ngrams)
    return 1.0 - len(unique) / len(ngrams)


# ============================================================================
# Main sweep
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sweep diffusion steps and block sizes")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--hellaswag_path", type=str,
                        default="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/Rowan/hellaswag/main")
    parser.add_argument("--output_dir", type=str, default="ablation_outputs/sweep_results")
    parser.add_argument("--block_sizes", type=str, default="4,8,16,32,64,128")
    parser.add_argument("--step_counts", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gumbel_temperature", type=float, default=0.7)
    parser.add_argument("--final_temperature", type=float, default=0.2)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    step_counts = [int(x) for x in args.step_counts.split(",")]

    device = f"cuda:{args.gpu_id}"
    torch.cuda.set_device(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device).eval()

    mask_token_id = model.config.mask_token_id

    # --- Load HellaSwag prompts ---
    print(f"Loading HellaSwag prompts...")
    try:
        ds = load_dataset(args.hellaswag_path, split="validation")
    except:
        ds = load_dataset("Rowan/hellaswag", split="validation")

    prompts = [ex["ctx"] for ex in ds]
    if args.num_samples and args.num_samples < len(prompts):
        import random
        random.seed(42)
        prompts = random.sample(prompts, args.num_samples)

    print(f"Using {len(prompts)} prompts")

    # --- Sweep ---
    all_results = []

    for block_size in block_sizes:
        for num_steps in step_counts:
            config_name = f"b{block_size}_s{num_steps}"
            print(f"\n{'='*60}")
            print(f"Config: block_size={block_size}, steps={num_steps}")
            print(f"{'='*60}")

            generations = []
            total_gen_tokens = 0
            total_time = 0.0

            for i in tqdm(range(0, len(prompts), args.batch_size), desc=config_name):
                batch_prompts = prompts[i:i + args.batch_size]

                # Tokenize with BOS
                inputs = tokenizer(
                    batch_prompts, return_tensors="pt",
                    padding=True, truncation=True, max_length=2560,
                    add_special_tokens=False,
                ).to(device)

                bos = torch.full(
                    (inputs["input_ids"].shape[0], 1),
                    tokenizer.bos_token_id,
                    dtype=inputs["input_ids"].dtype, device=device,
                )
                bos_mask = torch.ones(bos.shape, dtype=inputs["attention_mask"].dtype, device=device)
                input_ids = torch.cat([bos, inputs["input_ids"]], dim=1)
                attn_mask = torch.cat([bos_mask, inputs["attention_mask"]], dim=1)
                prompt_len = input_ids.shape[1]

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.inference_mode():
                    output_ids = generate_diffusion_quality(
                        model, input_ids, mask_token_id,
                        attention_mask=attn_mask,
                        max_new_tokens=args.max_new_tokens,
                        steps=num_steps,
                        block_size=block_size,
                        gumbel_temperature=args.gumbel_temperature,
                        final_temperature=args.final_temperature,
                    )

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                total_time += elapsed

                gen_ids = output_ids[:, prompt_len:]
                total_gen_tokens += gen_ids.numel()
                gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                generations.extend(gen_texts)

            # --- Compute metrics ---
            tokens_per_sec = total_gen_tokens / total_time if total_time > 0 else 0
            rep_rates = [compute_repetition_rate(g) for g in generations]
            avg_rep = sum(rep_rates) / len(rep_rates) if rep_rates else 0

            result = {
                "block_size": block_size,
                "num_steps": num_steps,
                "config_name": config_name,
                "tokens_per_sec": tokens_per_sec,
                "total_time_sec": total_time,
                "total_tokens": total_gen_tokens,
                "avg_repetition_4gram": avg_rep,
                "num_samples": len(generations),
                "max_new_tokens": args.max_new_tokens,
            }
            all_results.append(result)

            print(f"  Throughput: {tokens_per_sec:.1f} tok/s")
            print(f"  Avg 4-gram rep: {avg_rep:.4f}")

            # Save generations for this config (for later PPL calculation)
            gen_file = os.path.join(args.output_dir, f"{config_name}.jsonl")
            with open(gen_file, "w") as f:
                for j, gen in enumerate(generations):
                    json.dump({
                        "id": j,
                        "prompt": prompts[j],
                        "generation": gen,
                    }, f)
                    f.write("\n")

    # --- Save summary ---
    summary = {
        "model_path": args.model_path,
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "results": all_results,
    }
    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print summary table ---
    print(f"\n{'='*80}")
    print(f"SWEEP RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Block':>6} {'Steps':>6} {'Tok/s':>10} {'4gram-Rep':>10}")
    print(f"{'-'*42}")
    for r in all_results:
        print(f"{r['block_size']:>6} {r['num_steps']:>6} "
              f"{r['tokens_per_sec']:>10.1f} {r['avg_repetition_4gram']:>10.4f}")
    print(f"{'='*80}")
    print(f"Saved to: {summary_path}")
    print(f"Run PPL calculation separately on the generated .jsonl files.")


if __name__ == "__main__":
    main()
