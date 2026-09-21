#!/usr/bin/env python3
"""
分析序列长度 sweep 结果 — 输出 wall-clock time against sequence lengths 表格
可直接用于 rebuttal figure/table

用法: python3 analyze_seqlen_sweep.py
"""
import json
import os
import glob
import statistics

BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output"

MODELS = {
    "CARD":  "sweep_card_1b",
    "ARM":   "sweep_arm_1b",
    "MDLM":  "sweep_mdlm_1b",
    "BD3LM": "sweep_bd3lm_1b",
}

SEQ_LENS = [512, 1024, 2048, 4096]
SKIP_WARMUP = 50  # 跳过前50步 warmup

def find_trainer_state(model_dir):
    direct = os.path.join(model_dir, "trainer_state.json")
    if os.path.exists(direct):
        return direct
    ckpts = sorted(glob.glob(os.path.join(model_dir, "checkpoint-*/trainer_state.json")))
    if ckpts:
        return ckpts[-1]
    return None

def analyze_one(model_dir, seq_len):
    ts_file = find_trainer_state(model_dir)
    if not ts_file:
        return None

    d = json.load(open(ts_file))
    log = d.get("log_history", [])

    # 提取有 loss 的 step entries (logging_steps=10, 所以每10步一条)
    step_entries = [(e["step"], e) for e in log if "loss" in e and e.get("step", 0) > SKIP_WARMUP]
    if len(step_entries) < 2:
        return None

    # 方法1: 用 num_input_tokens_seen 差值算 tokens/step
    first = step_entries[0][1]
    last = step_entries[-1][1]
    steps_delta = last["step"] - first["step"]
    tokens_delta = last.get("num_input_tokens_seen", 0) - first.get("num_input_tokens_seen", 0)
    tokens_per_step = tokens_delta / steps_delta if steps_delta > 0 else 0

    # 方法2: 用 epoch 差值 + 数据集大小反推 wall-clock (不可靠)
    # 方法3: 用 trainer_state 的 train_runtime (只在最终 summary 中)
    summary_entries = [e for e in log if "train_runtime" in e]
    total_steps = d.get("global_step", 200)
    if summary_entries:
        total_runtime = summary_entries[-1]["train_runtime"]
        total_tokens = last.get("num_input_tokens_seen", 0)
        # 扣除 warmup 估算稳态 s/step
        # 稳态步数 = total_steps - SKIP_WARMUP
        steady_steps = total_steps - SKIP_WARMUP
        # 假设 warmup 步比稳态步慢 ~1.5x
        warmup_time_est = SKIP_WARMUP * (total_runtime / total_steps) * 1.2
        steady_time = total_runtime - warmup_time_est
        s_per_step = steady_time / steady_steps if steady_steps > 0 else total_runtime / total_steps
        tokens_per_s = tokens_per_step / s_per_step if s_per_step > 0 else 0
    else:
        # fallback: 简单用 total_runtime / total_steps
        s_per_step = None
        tokens_per_s = None

    # 方法4 (最可靠): 直接从 benchmark_train.sh 的 logging 间隔推算
    # 每隔 10 步 log 一次, 用连续 log 之间的 epoch 差反推
    # 但我们没有 wall-clock timestamp... 
    
    # 最终: 直接用 total_runtime / total_steps 作为 s/step
    if summary_entries:
        s_per_step_simple = summary_entries[-1]["train_runtime"] / total_steps
        tokens_per_s_simple = (last.get("num_input_tokens_seen", 0) / summary_entries[-1]["train_runtime"]) if summary_entries[-1]["train_runtime"] > 0 else 0
    else:
        s_per_step_simple = None
        tokens_per_s_simple = None

    return {
        "s_per_step": s_per_step_simple,
        "tokens_per_s": tokens_per_s_simple,
        "tokens_per_step": tokens_per_step,
        "total_flos": d.get("total_flos", 0),
        "total_steps": total_steps,
    }

# 收集所有结果
results = {}  # results[model][seq_len] = {...}
missing = []
for model_name, model_prefix in MODELS.items():
    results[model_name] = {}
    for seq_len in SEQ_LENS:
        model_dir = os.path.join(BASE, f"{model_prefix}_seq{seq_len}")
        r = analyze_one(model_dir, seq_len)
        if r:
            results[model_name][seq_len] = r
        else:
            missing.append(f"{model_name}/seq{seq_len}")

if missing:
    print(f"⚠️  Missing results: {', '.join(missing)}")
    print()

# 打印 s/step 表格
print("=" * 72)
print("Table R1: Training Time per Step (s/step) vs Sequence Length")
print("Hardware: 8× H800 GPUs, bs=4, ga=16, eff_batch=512")
print("=" * 72)

header = f"{'Model':<10}" + "".join(f"{'seq=' + str(s):>15}" for s in SEQ_LENS)
print(header)
print("-" * 72)

for model_name in MODELS:
    row = f"{model_name:<10}"
    for seq_len in SEQ_LENS:
        if seq_len in results[model_name] and results[model_name][seq_len]["s_per_step"] is not None:
            val = results[model_name][seq_len]["s_per_step"]
            row += f"{val:>15.3f}"
        else:
            row += f"{'—':>15}"
    print(row)

# 打印 tokens/s 表格
print()
print("=" * 72)
print("Table R2: Training Throughput (tokens/s) vs Sequence Length")
print("=" * 72)

header = f"{'Model':<10}" + "".join(f"{'seq=' + str(s):>15}" for s in SEQ_LENS)
print(header)
print("-" * 72)

for model_name in MODELS:
    row = f"{model_name:<10}"
    for seq_len in SEQ_LENS:
        if seq_len in results[model_name] and results[model_name][seq_len]["tokens_per_s"] is not None:
            val = results[model_name][seq_len]["tokens_per_s"]
            row += f"{val:>15,.0f}"
        else:
            row += f"{'—':>15}"
    print(row)

# 打印 FLOPs 表格
print()
print("=" * 72)
print("Table R3: Total FLOPs per Step (normalized)")
print("=" * 72)

header = f"{'Model':<10}" + "".join(f"{'seq=' + str(s):>15}" for s in SEQ_LENS)
print(header)
print("-" * 72)

for model_name in MODELS:
    row = f"{model_name:<10}"
    for seq_len in SEQ_LENS:
        if seq_len in results[model_name]:
            flos = results[model_name][seq_len]["total_flos"]
            steps = results[model_name][seq_len]["total_steps"]
            flos_per_step = flos / steps if steps > 0 else 0
            row += f"{flos_per_step:>15.2e}"
        else:
            row += f"{'—':>15}"
    print(row)

# Speedup vs ARM
print()
print("=" * 72)
print("Speedup: s/step ratio vs ARM (lower = faster)")
print("=" * 72)

header = f"{'Model':<10}" + "".join(f"{'seq=' + str(s):>15}" for s in SEQ_LENS)
print(header)
print("-" * 72)

for model_name in MODELS:
    row = f"{model_name:<10}"
    for seq_len in SEQ_LENS:
        if (seq_len in results[model_name] and seq_len in results.get("ARM", {})
                and results[model_name][seq_len]["s_per_step"] is not None
                and results["ARM"][seq_len]["s_per_step"] is not None):
            arm_sps = results["ARM"][seq_len]["s_per_step"]
            model_sps = results[model_name][seq_len]["s_per_step"]
            ratio = arm_sps / model_sps if model_sps > 0 else 0
            row += f"{ratio:>14.2f}×"
        else:
            row += f"{'—':>15}"
    print(row)

# CSV 格式输出（方便画图）
print()
print("=" * 72)
print("CSV (for plotting)")
print("=" * 72)
print("model,seq_len,s_per_step,tokens_per_s,flos_per_step")
for model_name in MODELS:
    for seq_len in SEQ_LENS:
        if seq_len in results[model_name]:
            r = results[model_name][seq_len]
            sps = r['s_per_step'] if r['s_per_step'] is not None else 0
            tps = r['tokens_per_s'] if r['tokens_per_s'] is not None else 0
            fps = r['total_flos'] / r['total_steps'] if r['total_steps'] > 0 else 0
            print(f"{model_name},{seq_len},{sps:.4f},{tps:.0f},{fps:.2e}")
