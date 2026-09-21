"""
Aggregate and visualize all ablation experiment results.

Reads results from all ablation experiments and produces:
1. LaTeX-ready tables for the paper
2. Matplotlib plots for key comparisons
3. JSON summary for programmatic access

Usage:
    python -m ablation.aggregate_results \
        --ablation1_dir ablation_outputs \
        --sweep_dir ablation_outputs/sweep_results \
        --efficiency_dir ablation_outputs/efficiency \
        --output_dir ablation_outputs/figures
"""

import argparse
import json
import os
import glob
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ============================================================================
# Ablation 1: Component decomposition bar chart
# ============================================================================

def plot_ablation1(results_dir: str, output_dir: str):
    """
    Plot bar chart comparing component variants.
    Expected: eval_results.json in each variant subdirectory.
    """
    variants = ["full_card", "no_daum", "no_tail_bias", "no_both", "prefix_lm"]
    labels = [
        "CARD\n(Full)",
        "w/o DAUM",
        "w/o Tail-Bias",
        "w/o Both",
        "Prefix-LM",
    ]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#95a5a6"]

    losses = []
    found = []

    for variant in variants:
        # Look for eval results in the checkpoint dirs
        variant_dir = os.path.join(results_dir, f"ablation1_{variant}")
        eval_path = os.path.join(variant_dir, "eval_results.json")
        
        # Also check inside the training output
        if not os.path.exists(eval_path):
            # Try to find trainer_state.json for eval_loss
            state_files = glob.glob(os.path.join(variant_dir, "checkpoint-*", "trainer_state.json"))
            if state_files:
                state_files.sort()
                with open(state_files[-1]) as f:
                    state = json.load(f)
                # Get last eval loss from log history
                eval_losses = [
                    entry.get("eval_loss") 
                    for entry in state.get("log_history", []) 
                    if "eval_loss" in entry
                ]
                if eval_losses:
                    losses.append(eval_losses[-1])
                    found.append(True)
                    continue
        elif os.path.exists(eval_path):
            with open(eval_path) as f:
                result = json.load(f)
            losses.append(result.get("eval_loss", float("nan")))
            found.append(True)
            continue

        losses.append(float("nan"))
        found.append(False)

    if not any(found):
        print("[Ablation 1] No results found. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(variants))
    bars = ax.bar(x, losses, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, loss, f in zip(bars, losses, found):
        if f and not np.isnan(loss):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{loss:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Evaluation Loss (NTP)", fontsize=13)
    ax.set_title("Ablation 1: Component Decomposition of CARD", fontsize=14, fontweight="bold")
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation1_components.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[Ablation 1] Saved: {path}")
    plt.close()


# ============================================================================
# Ablation 2 & 3: Steps and block size sweep heatmap + line plots
# ============================================================================

def plot_sweep_results(sweep_dir: str, output_dir: str):
    """
    Plot sweep results from sweep_summary.json.
    - Heatmap: block_size x steps -> tok/s
    - Line plot: steps -> quality metric
    """
    summary_path = os.path.join(sweep_dir, "sweep_summary.json")
    if not os.path.exists(summary_path):
        print(f"[Sweep] No results found at {summary_path}. Skipping.")
        return

    with open(summary_path) as f:
        data = json.load(f)

    results = data["results"]

    # --- Throughput heatmap ---
    block_sizes = sorted(set(r["block_size"] for r in results))
    step_counts = sorted(set(r["num_steps"] for r in results))

    if len(block_sizes) > 1 and len(step_counts) > 1:
        throughput_matrix = np.full((len(block_sizes), len(step_counts)), np.nan)
        rep_matrix = np.full((len(block_sizes), len(step_counts)), np.nan)

        for r in results:
            bi = block_sizes.index(r["block_size"])
            si = step_counts.index(r["num_steps"])
            throughput_matrix[bi, si] = r["tokens_per_sec"]
            rep_matrix[bi, si] = r.get("avg_repetition_4gram", np.nan)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Throughput
        im1 = axes[0].imshow(throughput_matrix, aspect='auto', cmap='YlOrRd')
        axes[0].set_xticks(range(len(step_counts)))
        axes[0].set_xticklabels(step_counts)
        axes[0].set_yticks(range(len(block_sizes)))
        axes[0].set_yticklabels(block_sizes)
        axes[0].set_xlabel("Diffusion Steps")
        axes[0].set_ylabel("Block Size")
        axes[0].set_title("Throughput (tokens/sec)")
        fig.colorbar(im1, ax=axes[0])

        for i in range(len(block_sizes)):
            for j in range(len(step_counts)):
                v = throughput_matrix[i, j]
                if not np.isnan(v):
                    axes[0].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)

        # Repetition
        im2 = axes[1].imshow(rep_matrix, aspect='auto', cmap='YlOrRd_r')
        axes[1].set_xticks(range(len(step_counts)))
        axes[1].set_xticklabels(step_counts)
        axes[1].set_yticks(range(len(block_sizes)))
        axes[1].set_yticklabels(block_sizes)
        axes[1].set_xlabel("Diffusion Steps")
        axes[1].set_ylabel("Block Size")
        axes[1].set_title("4-gram Repetition Rate (lower=better)")
        fig.colorbar(im2, ax=axes[1])

        for i in range(len(block_sizes)):
            for j in range(len(step_counts)):
                v = rep_matrix[i, j]
                if not np.isnan(v):
                    axes[1].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)

        plt.suptitle("CARD: Block Size × Diffusion Steps Trade-off", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(output_dir, "sweep_heatmap.pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[Sweep] Saved heatmap: {path}")
        plt.close()

    # --- Line plot: fixed block_size, vary steps ---
    for bs in block_sizes:
        subset = [r for r in results if r["block_size"] == bs]
        if len(subset) < 2:
            continue

        subset.sort(key=lambda x: x["num_steps"])
        steps_arr = [r["num_steps"] for r in subset]
        tps_arr = [r["tokens_per_sec"] for r in subset]
        rep_arr = [r.get("avg_repetition_4gram", 0) for r in subset]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = '#2196F3'
        ax1.plot(steps_arr, tps_arr, 'o-', color=color1, linewidth=2, markersize=8, label="Throughput")
        ax1.set_xlabel("Diffusion Steps", fontsize=12)
        ax1.set_ylabel("Tokens/sec", color=color1, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = '#F44336'
        ax2.plot(steps_arr, rep_arr, 's--', color=color2, linewidth=2, markersize=8, label="4-gram Rep")
        ax2.set_ylabel("4-gram Repetition Rate", color=color2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.set_title(f"Steps Sensitivity (block_size={bs})", fontsize=13, fontweight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax1.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, f"steps_sensitivity_b{bs}.pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[Sweep] Saved line plot: {path}")
        plt.close()


# ============================================================================
# Ablation 4: Efficiency comparison
# ============================================================================

def plot_efficiency(efficiency_dir: str, output_dir: str):
    """Plot efficiency comparison between CARD and AR."""
    bench_path = os.path.join(efficiency_dir, "efficiency_benchmark.json")
    if not os.path.exists(bench_path):
        print(f"[Efficiency] No results found at {bench_path}. Skipping.")
        return

    with open(bench_path) as f:
        data = json.load(f)

    results = data["results"]

    # --- Plot: Tokens/sec vs GenLen for batch_size=1 ---
    for target_bs in [1, 8]:
        ar_data = [r for r in results if r["method"] == "AR" and r["batch_size"] == target_bs]
        card_data_by_config = {}
        for r in results:
            if r["method"] == "CARD" and r["batch_size"] == target_bs:
                key = f"b{r['block_size']}_s{r['steps']}"
                if key not in card_data_by_config:
                    card_data_by_config[key] = []
                card_data_by_config[key].append(r)

        if not ar_data:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # AR baseline
        gen_lens = sorted(set(r["max_new_tokens"] for r in ar_data))
        ar_tps = []
        for gl in gen_lens:
            matching = [r for r in ar_data if r["max_new_tokens"] == gl]
            if matching:
                ar_tps.append(np.mean([r["tokens_per_sec"] for r in matching]))
            else:
                ar_tps.append(np.nan)
        ax.plot(gen_lens, ar_tps, 'k-o', linewidth=2.5, markersize=8, label="AR (Llama)", zorder=10)

        # CARD configs
        colors = plt.cm.tab10(np.linspace(0, 1, len(card_data_by_config)))
        for (config_name, config_results), color in zip(
            sorted(card_data_by_config.items()), colors
        ):
            c_gen_lens = sorted(set(r["max_new_tokens"] for r in config_results))
            c_tps = []
            for gl in c_gen_lens:
                matching = [r for r in config_results if r["max_new_tokens"] == gl]
                if matching:
                    c_tps.append(np.mean([r["tokens_per_sec"] for r in matching]))
                else:
                    c_tps.append(np.nan)
            ax.plot(c_gen_lens, c_tps, '-s', color=color, linewidth=1.5,
                    markersize=6, label=f"CARD ({config_name})")

        ax.set_xlabel("Generation Length (tokens)", fontsize=12)
        ax.set_ylabel("Throughput (tokens/sec)", fontsize=12)
        ax.set_title(f"Throughput Comparison: CARD vs AR (batch_size={target_bs})",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        path = os.path.join(output_dir, f"efficiency_bs{target_bs}.pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[Efficiency] Saved: {path}")
        plt.close()

    # --- FLOPs comparison bar chart ---
    # Pick one representative setting
    target_cfg = [r for r in results if r["prompt_len"] == 128 and r["max_new_tokens"] == 128 and r["batch_size"] == 1]
    if target_cfg:
        fig, ax = plt.subplots(figsize=(10, 5))
        methods = []
        flops = []
        tps = []
        for r in sorted(target_cfg, key=lambda x: x["est_flops_per_token"]):
            if r["method"] == "AR":
                label = "AR"
            else:
                label = f"CARD b={r['block_size']} s={r['steps']}"
            methods.append(label)
            flops.append(r["est_flops_per_token"])
            tps.append(r["tokens_per_sec"])

        x = np.arange(len(methods))
        width = 0.35
        bars1 = ax.bar(x - width/2, [f/1e9 for f in flops], width, label="FLOPs/token (G)", color="#3498db")
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, tps, width, label="Tokens/sec", color="#2ecc71", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("FLOPs per token (×10⁹)", fontsize=11)
        ax2.set_ylabel("Tokens/sec", fontsize=11)
        ax.set_title("FLOPs vs Throughput (Prompt=128, Gen=128, BS=1)", fontsize=12, fontweight="bold")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        path = os.path.join(output_dir, "flops_comparison.pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[Efficiency] Saved FLOPs chart: {path}")
        plt.close()


# ============================================================================
# LaTeX table generation
# ============================================================================

def generate_latex_tables(sweep_dir: str, output_dir: str):
    """Generate LaTeX-ready tables for the paper."""
    summary_path = os.path.join(sweep_dir, "sweep_summary.json")
    if not os.path.exists(summary_path):
        return

    with open(summary_path) as f:
        data = json.load(f)

    results = data["results"]

    # Table: Block size x Steps
    block_sizes = sorted(set(r["block_size"] for r in results))
    step_counts = sorted(set(r["num_steps"] for r in results))

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Generation throughput (tokens/sec) for different block sizes and diffusion steps.}")
    lines.append(r"\label{tab:sweep_throughput}")

    cols = "l" + "r" * len(step_counts)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    header = r"Block Size & " + " & ".join([f"{s} steps" for s in step_counts]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for bs in block_sizes:
        row = [str(bs)]
        for sc in step_counts:
            matching = [r for r in results if r["block_size"] == bs and r["num_steps"] == sc]
            if matching:
                row.append(f"{matching[0]['tokens_per_sec']:.0f}")
            else:
                row.append("--")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_path = os.path.join(output_dir, "sweep_table.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[LaTeX] Saved: {latex_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Aggregate ablation results")
    parser.add_argument("--ablation1_dir", type=str, default="ablation_outputs")
    parser.add_argument("--sweep_dir", type=str, default="ablation_outputs/sweep_results")
    parser.add_argument("--efficiency_dir", type=str, default="ablation_outputs/efficiency")
    parser.add_argument("--output_dir", type=str, default="ablation_outputs/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("AGGREGATING ABLATION RESULTS")
    print("=" * 60)

    plot_ablation1(args.ablation1_dir, args.output_dir)
    plot_sweep_results(args.sweep_dir, args.output_dir)
    plot_efficiency(args.efficiency_dir, args.output_dir)
    generate_latex_tables(args.sweep_dir, args.output_dir)

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
