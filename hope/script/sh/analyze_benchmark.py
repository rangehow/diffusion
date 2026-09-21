#!/usr/bin/env python3
"""
分析基准测试结果 — 从 trainer_state.json 提取 s/step 和 tokens/s
用法: python3 analyze_benchmark.py
"""
import json
import os
import glob

BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output"

MODELS = {
    "CARD":  "benchmark_card_1b",
    "ARM":   "benchmark_arm_1b",
    "MDLM":  "benchmark_mdlm_1b",
    "BD3LM": "benchmark_bd3lm_1b",
}

SKIP_WARMUP = 50  # 跳过前 50 步作为 warmup

def find_trainer_state(model_dir):
    """在 model_dir 下找 trainer_state.json（可能在 checkpoint-* 子目录）"""
    direct = os.path.join(model_dir, "trainer_state.json")
    if os.path.exists(direct):
        return direct
    ckpts = sorted(glob.glob(os.path.join(model_dir, "checkpoint-*/trainer_state.json")))
    if ckpts:
        return ckpts[-1]
    return None

def analyze(name, path):
    ts_file = find_trainer_state(path)
    if not ts_file:
        return {"name": name, "error": "trainer_state.json not found"}

    d = json.load(open(ts_file))
    log = d.get("log_history", [])
    
    # 找到 warmup 后的 entries（跳过前 SKIP_WARMUP 步）
    entries = [e for e in log if e.get("step", 0) > SKIP_WARMUP and "train_runtime" in e]
    
    if len(entries) < 2:
        return {"name": name, "error": f"Not enough log entries after warmup (got {len(entries)})"}
    
    # 用 step 100 和最后一个 step 计算稳态 s/step
    e_start = entries[0]
    e_end = entries[-1]
    
    steps_delta = e_end["step"] - e_start["step"]
    time_delta = e_end["train_runtime"] - e_start["train_runtime"]
    tokens_delta = e_end["num_input_tokens_seen"] - e_start["num_input_tokens_seen"]
    
    s_per_step = time_delta / steps_delta if steps_delta > 0 else float("inf")
    tokens_per_s = tokens_delta / time_delta if time_delta > 0 else 0
    tok_per_step = tokens_delta / steps_delta if steps_delta > 0 else 0
    
    # 从 training_args.json 提取配置（如果有）
    args_file = os.path.join(os.path.dirname(ts_file), "..", "training_args.json")
    if not os.path.exists(args_file):
        args_file = os.path.join(os.path.dirname(ts_file), "training_args.json")
    
    bs = ga = mode = "?"
    if os.path.exists(args_file):
        args = json.load(open(args_file))
        bs = args.get("per_device_train_batch_size", "?")
        ga = args.get("gradient_accumulation_steps", "?")
        mode = args.get("mode", "?")
    
    return {
        "name": name,
        "mode": mode,
        "bs": bs,
        "ga": ga,
        "eff_batch": f"{bs}×{ga}×8={bs*ga*8}" if isinstance(bs, int) else "?",
        "s_per_step": s_per_step,
        "tok_per_step": tok_per_step,
        "tokens_per_s": tokens_per_s,
        "measured_steps": f"{e_start['step']}-{e_end['step']}",
        "total_flos": d.get("total_flos", 0),
    }

print("=" * 90)
print(f"{'Model':<8} {'Mode':<6} {'BS×GA×8':<16} {'s/step':>8} {'tok/step':>12} {'tok/s':>12} {'Steps':>12}")
print("=" * 90)

results = []
for name, subdir in MODELS.items():
    path = os.path.join(BASE, subdir)
    r = analyze(name, path)
    results.append(r)
    
    if "error" in r:
        print(f"{name:<8} ERROR: {r['error']}")
    else:
        print(f"{r['name']:<8} {r['mode']:<6} {r['eff_batch']:<16} {r['s_per_step']:>8.2f} {r['tok_per_step']:>12,.0f} {r['tokens_per_s']:>12,.0f} {r['measured_steps']:>12}")

print("=" * 90)

# 对比表（归一化到 CARD）
card = next((r for r in results if r["name"] == "CARD" and "error" not in r), None)
if card:
    print(f"\n相对于 CARD 的速度比:")
    for r in results:
        if "error" not in r:
            ratio = card["s_per_step"] / r["s_per_step"]
            tok_ratio = r["tokens_per_s"] / card["tokens_per_s"]
            print(f"  {r['name']:<8}: {ratio:.2f}× step speed, {tok_ratio:.2f}× token throughput")

print("\n注意: 由于 GA 步数不同，s/step 跨模型不直接可比。")
print("      tokens/s 是最公平的吞吐量度量。")
