"""
Generation Metrics Report
Combines PPL and Diversity metrics into unified reports and visualizations.
"""

import os
import json
import argparse
import glob
from datetime import datetime
from typing import Dict, List, Optional
import math


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ppl_results(gen_dir: str) -> Dict[str, Dict]:
    """Load all PPL result files from directory."""
    ppl_results = {}

    for filepath in glob.glob(os.path.join(gen_dir, "ppl_*.json")):
        with open(filepath, "r") as f:
            data = json.load(f)

        model_name = data.get("ppl_model_name", os.path.basename(filepath))
        ppl_results[model_name] = data.get("results", {})

    return ppl_results


def load_diversity_results(gen_dir: str) -> Dict[str, Dict]:
    """Load diversity metrics from directory."""
    filepath = os.path.join(gen_dir, "diversity_metrics.json")

    if not os.path.exists(filepath):
        return {}

    with open(filepath, "r") as f:
        data = json.load(f)

    return data.get("results", {})


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_markdown_report(
    ppl_results: Dict[str, Dict],
    diversity_results: Dict[str, Dict],
    output_path: str,
    title: str = "Generation Evaluation Report",
):
    """Generate a comprehensive markdown report."""

    # Get all method names
    all_methods = set()
    for results in ppl_results.values():
        all_methods.update(results.keys())
    all_methods.update(diversity_results.keys())
    all_methods = sorted(all_methods)

    lines = []
    lines.append(f"# {title}")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # ========== Summary Table ==========
    lines.append("## Summary Table\n")

    # Build header
    header_parts = ["Method"]
    ppl_models = sorted(ppl_results.keys())
    for model in ppl_models:
        header_parts.append(f"PPL ({model})")
    header_parts.extend(["Dist-1", "Dist-2", "Rep-2", "Rep-3", "SeqRep-2", "AvgLen"])

    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")

    # Build rows
    for method in all_methods:
        row = [method]

        # PPL columns
        for model in ppl_models:
            if method in ppl_results.get(model, {}):
                ppl = ppl_results[model][method].get("ppl_mean", float('inf'))
                row.append(f"{ppl:.2f}" if ppl < 1e5 else "N/A")
            else:
                row.append("N/A")

        # Diversity columns
        if method in diversity_results:
            div = diversity_results[method]
            row.append(f"{div.get('corpus_distinct_1', 0):.4f}")
            row.append(f"{div.get('corpus_distinct_2', 0):.4f}")
            row.append(f"{div.get('rep_2', 0):.4f}")
            row.append(f"{div.get('rep_3', 0):.4f}")
            row.append(f"{div.get('seq_rep_2', 0):.4f}")
            row.append(f"{div.get('avg_length', 0):.1f}")
        else:
            row.extend(["N/A"] * 6)

        lines.append("| " + " | ".join(row) + " |")

    # ========== Interpretation ==========
    lines.append("\n## Metric Interpretation\n")
    lines.append("| Metric | Description | Optimal |")
    lines.append("| --- | --- | --- |")
    lines.append("| PPL | Fluency/naturalness (lower = more predictable) | Lower, but watch for repetition |")
    lines.append("| Dist-1/2 | Vocabulary diversity (unique n-grams ratio) | Higher |")
    lines.append("| Rep-2/3 | N-gram repetition rate | Lower |")
    lines.append("| SeqRep-2 | Consecutive repetition (stuttering) | Lower |")

    # ========== Quality-Diversity Trade-off ==========
    lines.append("\n## Quality-Diversity Trade-off\n")
    lines.append("```")
    lines.append("                    High Diversity")
    lines.append("                          │")
    lines.append("            ┌─────────────┼─────────────┐")
    lines.append("            │   Diverse   │   Ideal     │")
    lines.append("            │  but noisy  │  (target)   │")
    lines.append("  High PPL ─┼─────────────┼─────────────┼─ Low PPL")
    lines.append("            │    Bad      │  Repetitive │")
    lines.append("            │  (garbage)  │  but fluent │")
    lines.append("            └─────────────┼─────────────┘")
    lines.append("                          │")
    lines.append("                    Low Diversity")
    lines.append("```")

    # ========== Per-Method Analysis ==========
    lines.append("\n## Per-Method Analysis\n")

    for method in all_methods:
        lines.append(f"### {method}\n")

        # PPL info
        for model in ppl_models:
            if method in ppl_results.get(model, {}):
                data = ppl_results[model][method]
                lines.append(f"**PPL ({model}):** {data.get('ppl_mean', 0):.2f} ± {data.get('ppl_std', 0):.2f}")

        # Diversity info
        if method in diversity_results:
            div = diversity_results[method]
            lines.append(f"\n**Diversity Metrics:**")
            lines.append(f"- Corpus Distinct-1/2/3: {div.get('corpus_distinct_1', 0):.4f} / {div.get('corpus_distinct_2', 0):.4f} / {div.get('corpus_distinct_3', 0):.4f}")
            lines.append(f"- Rep-2/3/4: {div.get('rep_2', 0):.4f} / {div.get('rep_3', 0):.4f} / {div.get('rep_4', 0):.4f}")
            lines.append(f"- Vocabulary Size: {div.get('vocab_size', 0)}, Entropy: {div.get('token_entropy', 0):.2f} bits")

        lines.append("")

    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def generate_csv_report(
    ppl_results: Dict[str, Dict],
    diversity_results: Dict[str, Dict],
    output_path: str,
):
    """Generate CSV for easy import to Excel/Sheets."""

    all_methods = set()
    for results in ppl_results.values():
        all_methods.update(results.keys())
    all_methods.update(diversity_results.keys())
    all_methods = sorted(all_methods)

    ppl_models = sorted(ppl_results.keys())

    lines = []

    # Header
    header = ["method"]
    for model in ppl_models:
        header.extend([f"ppl_{model}", f"ppl_std_{model}"])
    header.extend([
        "corpus_distinct_1", "corpus_distinct_2", "corpus_distinct_3",
        "sample_distinct_1", "sample_distinct_2", "sample_distinct_3",
        "rep_2", "rep_3", "rep_4",
        "seq_rep_2", "seq_rep_3",
        "avg_length", "vocab_size", "token_entropy"
    ])
    lines.append(",".join(header))

    # Data rows
    for method in all_methods:
        row = [method]

        # PPL
        for model in ppl_models:
            if method in ppl_results.get(model, {}):
                data = ppl_results[model][method]
                row.append(f"{data.get('ppl_mean', ''):.4f}")
                row.append(f"{data.get('ppl_std', ''):.4f}")
            else:
                row.extend(["", ""])

        # Diversity
        if method in diversity_results:
            div = diversity_results[method]
            row.extend([
                f"{div.get('corpus_distinct_1', ''):.4f}",
                f"{div.get('corpus_distinct_2', ''):.4f}",
                f"{div.get('corpus_distinct_3', ''):.4f}",
                f"{div.get('sample_distinct_1', ''):.4f}",
                f"{div.get('sample_distinct_2', ''):.4f}",
                f"{div.get('sample_distinct_3', ''):.4f}",
                f"{div.get('rep_2', ''):.4f}",
                f"{div.get('rep_3', ''):.4f}",
                f"{div.get('rep_4', ''):.4f}",
                f"{div.get('seq_rep_2', ''):.4f}",
                f"{div.get('seq_rep_3', ''):.4f}",
                f"{div.get('avg_length', ''):.1f}",
                f"{div.get('vocab_size', '')}",
                f"{div.get('token_entropy', ''):.4f}",
            ])
        else:
            row.extend([""] * 14)

        lines.append(",".join(row))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def generate_ascii_chart(
    ppl_results: Dict[str, Dict],
    diversity_results: Dict[str, Dict],
    ppl_model: Optional[str] = None,
) -> str:
    """Generate ASCII scatter plot of PPL vs Diversity."""

    if not ppl_model and ppl_results:
        ppl_model = list(ppl_results.keys())[0]

    if not ppl_model or ppl_model not in ppl_results:
        return "No PPL data available for chart."

    # Collect data points
    points = []
    for method in diversity_results:
        if method in ppl_results.get(ppl_model, {}):
            ppl = ppl_results[ppl_model][method].get("ppl_mean", float('inf'))
            div = diversity_results[method].get("corpus_distinct_2", 0)
            if ppl < 1e5:
                points.append((method, ppl, div))

    if not points:
        return "No overlapping data for chart."

    # Chart dimensions
    width = 60
    height = 20

    # Calculate ranges
    ppls = [p[1] for p in points]
    divs = [p[2] for p in points]

    ppl_min, ppl_max = min(ppls), max(ppls)
    div_min, div_max = min(divs), max(divs)

    # Add margins
    ppl_range = ppl_max - ppl_min or 1
    div_range = div_max - div_min or 0.1

    ppl_min -= ppl_range * 0.1
    ppl_max += ppl_range * 0.1
    div_min -= div_range * 0.1
    div_max += div_range * 0.1

    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot points
    labels = []
    for i, (method, ppl, div) in enumerate(points):
        x = int((ppl - ppl_min) / (ppl_max - ppl_min) * (width - 1))
        y = int((div - div_min) / (div_max - div_min) * (height - 1))
        y = height - 1 - y  # Flip y-axis

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        marker = chr(ord('A') + i) if i < 26 else '*'
        grid[y][x] = marker
        labels.append(f"  {marker}: {method} (PPL={ppl:.1f}, Dist2={div:.4f})")

    # Build output
    lines = []
    lines.append(f"\n  PPL vs Corpus Distinct-2 (PPL model: {ppl_model})")
    lines.append("  " + "=" * (width + 4))

    # Y-axis label
    lines.append(f"  Dist-2")
    lines.append(f"  {div_max:.3f} ┐")

    for i, row in enumerate(grid):
        if i == height // 2:
            lines.append(f"         │{''.join(row)}│")
        else:
            lines.append(f"         │{''.join(row)}│")

    lines.append(f"  {div_min:.3f} ┴" + "─" * width + "┘")
    lines.append(f"         {ppl_min:.1f}" + " " * (width - 10) + f"{ppl_max:.1f}")
    lines.append(f"                        PPL →")
    lines.append("")
    lines.append("  Legend:")
    lines.extend(labels)
    lines.append("")
    lines.append("  ★ Ideal region: Lower-left (low PPL, high diversity)")

    return "\n".join(lines)


def print_terminal_report(
    ppl_results: Dict[str, Dict],
    diversity_results: Dict[str, Dict],
):
    """Print a nicely formatted report to terminal."""

    all_methods = set()
    for results in ppl_results.values():
        all_methods.update(results.keys())
    all_methods.update(diversity_results.keys())
    all_methods = sorted(all_methods)

    ppl_models = sorted(ppl_results.keys())

    print("\n" + "=" * 100)
    print("GENERATION EVALUATION REPORT")
    print("=" * 100)

    # Determine column widths
    method_width = max(15, max(len(m) for m in all_methods) + 2)

    # Header
    header = f"{'Method':<{method_width}}"
    for model in ppl_models:
        short_name = model[:10] if len(model) > 10 else model
        header += f" {'PPL-'+short_name:<12}"
    header += f" {'Dist-1':<8} {'Dist-2':<8} {'Rep-2':<8} {'Rep-3':<8} {'SeqR-2':<8}"

    print(header)
    print("-" * len(header))

    # Data rows
    for method in all_methods:
        row = f"{method:<{method_width}}"

        # PPL columns
        for model in ppl_models:
            if method in ppl_results.get(model, {}):
                ppl = ppl_results[model][method].get("ppl_mean", float('inf'))
                row += f" {ppl:<12.2f}" if ppl < 1e5 else f" {'N/A':<12}"
            else:
                row += f" {'N/A':<12}"

        # Diversity columns
        if method in diversity_results:
            div = diversity_results[method]
            row += f" {div.get('corpus_distinct_1', 0):<8.4f}"
            row += f" {div.get('corpus_distinct_2', 0):<8.4f}"
            row += f" {div.get('rep_2', 0):<8.4f}"
            row += f" {div.get('rep_3', 0):<8.4f}"
            row += f" {div.get('seq_rep_2', 0):<8.4f}"
        else:
            row += f" {'N/A':<8}" * 5

        print(row)

    print("-" * len(header))

    # Print ASCII chart
    if ppl_results and diversity_results:
        print(generate_ascii_chart(ppl_results, diversity_results))

    # Interpretation
    print("\n" + "=" * 100)
    print("QUICK INTERPRETATION")
    print("=" * 100)
    print("  • Look for methods with LOW PPL + HIGH Distinct + LOW Rep")
    print("  • High PPL + High Distinct = diverse but disfluent")
    print("  • Low PPL + Low Distinct = fluent but repetitive (common failure mode)")
    print("  • High Rep-2/3 = model is 'looping' or 'stuttering'")
    print("=" * 100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate combined PPL + Diversity evaluation report"
    )

    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="Directory containing PPL and diversity metric results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as gen_dir)"
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["terminal", "markdown", "csv"],
        choices=["terminal", "markdown", "csv", "all"],
        help="Output format(s)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Generation Evaluation Report",
        help="Report title"
    )

    args = parser.parse_args()

    gen_dir = args.gen_dir
    output_dir = args.output_dir or gen_dir
    os.makedirs(output_dir, exist_ok=True)

    formats = args.format
    if "all" in formats:
        formats = ["terminal", "markdown", "csv"]

    # Load results
    print("📂 Loading results...")
    ppl_results = load_ppl_results(gen_dir)
    diversity_results = load_diversity_results(gen_dir)

    print(f"   PPL results: {list(ppl_results.keys()) or 'None found'}")
    print(f"   Diversity results: {'Found' if diversity_results else 'Not found'}")

    if not ppl_results and not diversity_results:
        print("❌ No results found! Run PPL and diversity calculators first.")
        return

    # Generate reports
    outputs = []

    if "terminal" in formats:
        print_terminal_report(ppl_results, diversity_results)

    if "markdown" in formats:
        md_path = os.path.join(output_dir, "evaluation_report.md")
        generate_markdown_report(ppl_results, diversity_results, md_path, args.title)
        outputs.append(md_path)
        print(f"\n✅ Markdown report: {md_path}")

    if "csv" in formats:
        csv_path = os.path.join(output_dir, "evaluation_report.csv")
        generate_csv_report(ppl_results, diversity_results, csv_path)
        outputs.append(csv_path)
        print(f"✅ CSV report: {csv_path}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()