"""
Per-model Gen PPL reporting (Reviewer 2Bg4).

The reviewer objects to reporting the mean of PPL from 4 reference models.
This script runs PPL calculation using each reference model SEPARATELY
and reports results in a table format.

Usage:
    python -m ablation.ppl_per_model \
        --gen_dir ppl/generation_outputs \
        --output_dir ablation_outputs/ppl_per_model \
        --batch_size 4

Reference models are hardcoded below; modify as needed.
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict

import torch

# Import from existing PPL calculation code
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ppl.ppl_calculation import (
    load_generations,
    calculate_ppl_multi_gpu,
    calculate_ppl_single_gpu,
    extract_model_name,
)


# ============================================================================
# Reference models for PPL evaluation
# ============================================================================

REFERENCE_MODELS = {
    "gpt2-large": "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/openai-community/gpt2-large/main",
    "llama-3.2-1B": "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/meta-llama/Llama-3.2-1B/main",
    "qwen2.5-1.5B": "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2.5-1.5B/main",
    "gemma-2-2B": "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/google/gemma-2-2b/main",
}


def main():
    parser = argparse.ArgumentParser(description="Per-model Gen PPL calculation")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Directory containing .jsonl generation files")
    parser.add_argument("--output_dir", type=str, default="ablation_outputs/ppl_per_model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_gpus", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    num_gpus = args.num_gpus or torch.cuda.device_count()

    # Load generations
    print("Loading generations...")
    generation_data, meta_data = load_generations(args.gen_dir)

    if not generation_data:
        print("No generation files found!")
        return

    print(f"Found {len(generation_data)} generation methods:")
    for name, samples in generation_data.items():
        print(f"  - {name}: {len(samples)} samples")

    # ========================================
    # Run PPL for each reference model
    # ========================================
    all_ppl_results = {}  # {ref_model_name: {gen_method: ppl_stats}}

    for ref_name, ref_path in REFERENCE_MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating with reference model: {ref_name}")
        print(f"Path: {ref_path}")
        print(f"{'='*60}")

        if not os.path.exists(ref_path):
            print(f"  WARNING: Model not found at {ref_path}, skipping.")
            continue

        model_results = {}

        for method_name, samples in generation_data.items():
            print(f"\n  [{method_name}]")
            texts = [s.get("prompt", "") + s.get("generation", "") for s in samples]

            if num_gpus > 1:
                ppls = calculate_ppl_multi_gpu(
                    ppl_model_path=ref_path,
                    texts=texts,
                    num_gpus=num_gpus,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
            else:
                ppls = calculate_ppl_single_gpu(
                    ppl_model_path=ref_path,
                    texts=texts,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )

            valid_ppls = [p for p in ppls if p != float('inf') and p < 1e6]
            mean_ppl = sum(valid_ppls) / len(valid_ppls) if valid_ppls else float('inf')
            std_ppl = (sum((p - mean_ppl) ** 2 for p in valid_ppls) / len(valid_ppls)) ** 0.5 if valid_ppls else 0

            model_results[method_name] = {
                "mean": mean_ppl,
                "std": std_ppl,
                "num_valid": len(valid_ppls),
                "num_total": len(ppls),
            }
            print(f"  PPL({ref_name}): {mean_ppl:.2f} ± {std_ppl:.2f}")

        all_ppl_results[ref_name] = model_results

    # ========================================
    # Save detailed results
    # ========================================
    output_data = {
        "gen_dir": args.gen_dir,
        "timestamp": datetime.now().isoformat(),
        "reference_models": list(REFERENCE_MODELS.keys()),
        "results": all_ppl_results,
    }

    json_path = os.path.join(args.output_dir, "ppl_per_model.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # ========================================
    # Print formatted table
    # ========================================
    gen_methods = sorted(generation_data.keys())
    ref_models = [r for r in REFERENCE_MODELS.keys() if r in all_ppl_results]

    print(f"\n{'='*100}")
    print(f"PER-MODEL GEN PPL RESULTS")
    print(f"{'='*100}")

    # Header
    header = f"{'Method':<30}"
    for ref in ref_models:
        header += f" {ref:>15}"
    header += f" {'Mean':>10}"
    print(header)
    print("-" * 100)

    for method in gen_methods:
        row = f"{method:<30}"
        ppls_for_avg = []
        for ref in ref_models:
            if ref in all_ppl_results and method in all_ppl_results[ref]:
                val = all_ppl_results[ref][method]["mean"]
                row += f" {val:>15.2f}"
                ppls_for_avg.append(val)
            else:
                row += f" {'--':>15}"
        
        if ppls_for_avg:
            avg = sum(ppls_for_avg) / len(ppls_for_avg)
            row += f" {avg:>10.2f}"
        else:
            row += f" {'--':>10}"
        print(row)

    print(f"{'='*100}")

    # ========================================
    # Generate LaTeX table
    # ========================================
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Generation PPL evaluated by each reference model separately.}")
    latex_lines.append(r"\label{tab:ppl_per_model}")

    cols = "l" + "c" * len(ref_models) + "c"
    latex_lines.append(r"\begin{tabular}{" + cols + "}")
    latex_lines.append(r"\toprule")
    
    header_tex = "Method"
    for ref in ref_models:
        header_tex += f" & {ref}"
    header_tex += r" & Mean \\"
    latex_lines.append(header_tex)
    latex_lines.append(r"\midrule")

    for method in gen_methods:
        row_tex = method.replace("_", r"\_")
        ppls_for_avg = []
        for ref in ref_models:
            if ref in all_ppl_results and method in all_ppl_results[ref]:
                val = all_ppl_results[ref][method]["mean"]
                row_tex += f" & {val:.2f}"
                ppls_for_avg.append(val)
            else:
                row_tex += " & --"
        if ppls_for_avg:
            avg = sum(ppls_for_avg) / len(ppls_for_avg)
            row_tex += f" & {avg:.2f}"
        else:
            row_tex += " & --"
        row_tex += r" \\"
        latex_lines.append(row_tex)

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    latex_path = os.path.join(args.output_dir, "ppl_per_model.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
