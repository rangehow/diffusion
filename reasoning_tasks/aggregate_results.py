"""
Aggregate evaluation results from reasoning tasks into a formatted table.

Usage:
    python -m reasoning_tasks.aggregate_results --results_dir reasoning_tasks/eval_output
"""

import argparse
import json
import os
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="reasoning_tasks/eval_output")
    args = parser.parse_args()
    
    # Collect all result files
    result_files = glob.glob(os.path.join(args.results_dir, "*.json"))
    
    if not result_files:
        print(f"No result files found in {args.results_dir}")
        return
    
    # Parse results
    results = {}
    for f in sorted(result_files):
        try:
            with open(f) as fh:
                data = json.load(fh)
            task = data["task"]
            mode = data["mode"].replace("reasoning_", "")
            accuracy = data["accuracy"]
            n_total = data["n_total"]
            
            if task not in results:
                results[task] = {}
            results[task][mode] = {
                "accuracy": accuracy,
                "n_correct": data["n_correct"],
                "n_total": n_total,
                "throughput": data.get("throughput_tok_per_sec", None),
                "gen_time": data.get("gen_time_seconds", None),
                "tokens": data.get("tokens_generated", None),
            }
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    # Format table
    tasks = sorted(results.keys())
    modes = sorted(set(m for t in results.values() for m in t.keys()))
    
    # ---- Accuracy Table ----
    print("\n## Reasoning Task Results (Exact-Match Accuracy)\n")
    header = "| Task | " + " | ".join(f"{m.upper()}" for m in modes) + " |"
    separator = "|------|" + "|".join("------" for _ in modes) + "|"
    print(header)
    print(separator)
    
    for task in tasks:
        row = f"| {task.upper()} |"
        for mode in modes:
            if mode in results[task]:
                acc = results[task][mode]["accuracy"]
                row += f" {acc*100:.1f}% |"
            else:
                row += " - |"
        print(row)
    
    # Average row
    row = "| **AVG** |"
    for mode in modes:
        accs = [results[t][mode]["accuracy"] for t in tasks if mode in results[t]]
        if accs:
            avg = sum(accs) / len(accs)
            row += f" **{avg*100:.1f}%** |"
        else:
            row += " - |"
    print(row)
    
    # ---- Throughput Table ----
    has_throughput = any(
        results[t][m].get("throughput") is not None
        for t in tasks for m in results[t]
    )
    
    if has_throughput:
        print("\n## Generation Throughput (tok/s)\n")
        header = "| Task | " + " | ".join(f"{m.upper()}" for m in modes) + " |"
        print(header)
        print(separator)
        
        for task in tasks:
            row = f"| {task.upper()} |"
            for mode in modes:
                if mode in results[task] and results[task][mode]["throughput"] is not None:
                    tp = results[task][mode]["throughput"]
                    row += f" {tp:,.0f} |"
                else:
                    row += " - |"
            print(row)
        
        # Average throughput row
        row = "| **AVG** |"
        for mode in modes:
            tps = [results[t][mode]["throughput"] for t in tasks 
                   if mode in results[t] and results[t][mode]["throughput"] is not None]
            if tps:
                avg = sum(tps) / len(tps)
                row += f" **{avg:,.0f}** |"
            else:
                row += " - |"
        print(row)
    
    print()
    
    # ---- LaTeX table (accuracy + throughput combined) ----
    print("% LaTeX table:")
    print("\\begin{tabular}{l" + "c" * len(modes) + "}")
    print("\\toprule")
    print("Task & " + " & ".join(f"\\textbf{{{m.upper()}}}" for m in modes) + " \\\\")
    print("\\midrule")
    print("\\multicolumn{" + str(len(modes)+1) + "}{c}{\\textit{Accuracy (\\%)}} \\\\")
    print("\\midrule")
    for task in tasks:
        row = f"{task.upper()}"
        for mode in modes:
            if mode in results[task]:
                acc = results[task][mode]["accuracy"]
                row += f" & {acc*100:.1f}"
            else:
                row += " & --"
        row += " \\\\"
        print(row)
    
    # Average
    row = "\\textbf{AVG}"
    for mode in modes:
        accs = [results[t][mode]["accuracy"] for t in tasks if mode in results[t]]
        if accs:
            avg = sum(accs) / len(accs)
            row += f" & \\textbf{{{avg*100:.1f}}}"
        else:
            row += " & --"
    row += " \\\\"
    print(row)
    
    if has_throughput:
        print("\\midrule")
        print("\\multicolumn{" + str(len(modes)+1) + "}{c}{\\textit{Throughput (tok/s)}} \\\\")
        print("\\midrule")
        for task in tasks:
            row = f"{task.upper()}"
            for mode in modes:
                if mode in results[task] and results[task][mode]["throughput"] is not None:
                    tp = results[task][mode]["throughput"]
                    row += f" & {tp:,.0f}"
                else:
                    row += " & --"
            row += " \\\\"
            print(row)
        
        row = "\\textbf{AVG}"
        for mode in modes:
            tps = [results[t][mode]["throughput"] for t in tasks 
                   if mode in results[t] and results[t][mode]["throughput"] is not None]
            if tps:
                avg = sum(tps) / len(tps)
                row += f" & \\textbf{{{avg:,.0f}}}"
            else:
                row += " & --"
        row += " \\\\"
        print(row)
    
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
