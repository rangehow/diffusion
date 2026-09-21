#!/bin/bash
# ============================================================
# Submit all 6 Qwen3-8B CPT experiments via Megatron-Bridge
#
# === 500B-TOKEN TRAINING (NO REPETITION) ===
#
# Hyperparameters (derived from SmolLM3 Stage 1 config):
#   LR:          1e-4 (SmolLM3 uses 2e-4 from scratch; CPT = 1/2)
#   Min LR:      1e-5 (1/10 of peak)
#   Warmup:      2000 steps (same as SmolLM3)
#   Seq length:  4096 (match SmolLM3)
#   GBS:         512 seqs × 4096 = 2.10M tokens/step
#   Iters:       238K (= ~500B tokens)
#   Optimizer:   AdamW (β1=0.9, β2=0.95, wd=0.1)
#   Schedule:    Cosine decay
#
# Data (no langid filtering needed — DCLM-dedup is 100% English):
#   All of stack_edu_python + nemotron_cc_math; dclm-dedup downsampled.
#   No source exceeds 1 epoch — zero token repetition.
#   dclm-dedup (downsampled):  84.26%  (421.3B of 750B available)
#   nemotron-cc-math (all):    12.32%  (61.6B — full dataset)
#   stack-edu-python (all):     3.42%  (17.1B — full dataset)
#
# Cluster: 4 nodes × 8 H800 = 32 GPUs, TP=4, DP=8
# Estimated wall time: 30–45 days at 32 GPUs
#   → Consider scaling to 8 nodes (64 GPUs) to halve this
#
# Experiments:
#   1. arm           — Pure ARM baseline (NTP)
#   2. card_naive    — CARD V1: scattered mask + DAUM
#   3. card_v3       — CARD V3: topological reorder + logical pos IDs (RoPE)
#   4. card_v4       — CARD V4: topological reorder + sequential pos IDs
#   5. adapter_only  — Suffix mask + LoRA adapter (freeze backbone)
#   6. adapter_full  — Suffix mask + LoRA adapter (train everything)
#
# Usage:
#   cd /path/to/diffusion/hope/script
#   bash submit_qwen3_cpt_megatron.sh
# ============================================================

set -e

# ─────── Data Config ───────
ARROW_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/datasets/qwen3_cpt_arrow"

# Multi-source JSON — 500B token mixture (no repetition)
# Use all of math + code, downsample dclm to fill remaining budget
HF_DATA_SOURCES='[
  {"path":"'"${ARROW_ROOT}/dclm_dedup"'",       "weight":0.8426, "name":"dclm_dedup"},
  {"path":"'"${ARROW_ROOT}/nemotron_cc_math"'",  "weight":0.1232, "name":"nemotron_math"},
  {"path":"'"${ARROW_ROOT}/stack_edu_python"'",  "weight":0.0342, "name":"stack_edu_python"}
]'

# ─────── Paths ───────
BASE_OUTPUT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/qwen3_cpt_megatron"
HOPE_TEMPLATE="../hope/qwen3_cpt_megatron_m17.hope"
BRIDGE_SCRIPT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/Megatron-Bridge/CARD_SCRIPT/qwen3_cpt"
LOG_FILE="./qwen3_cpt_megatron_submission.log"

echo "=== Qwen3-8B CPT — 500B tokens — Megatron-Bridge ===" | tee "$LOG_FILE"
echo "Arrow root: ${ARROW_ROOT}" | tee -a "$LOG_FILE"
echo "Output:     ${BASE_OUTPUT}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ─────── Verify data exists ───────
for src in dclm_dedup nemotron_cc_math stack_edu_python; do
    if [ ! -d "${ARROW_ROOT}/${src}" ]; then
        echo "ERROR: Arrow dataset not found: ${ARROW_ROOT}/${src}" | tee -a "$LOG_FILE"
        echo "Run step0_prepare_data.py first:" | tee -a "$LOG_FILE"
        echo "  python3 ${BRIDGE_SCRIPT_DIR}/step0_prepare_data.py --output ${ARROW_ROOT}" | tee -a "$LOG_FILE"
        exit 1
    fi
done
echo "All 3 data sources verified ✓" | tee -a "$LOG_FILE"

# ─────── Experiment Definitions ───────
experiments=(
    "arm|"
    "card_naive|--tail-factor 1.5 --decay 0.5 --beta 1.0"
    "card_v3|--tail-factor 1.5"
    "card_v4|--tail-factor 1.5"
    "adapter_only|--adapter-rank 64 --alpha 1.0"
    "adapter_full|--adapter-rank 64 --alpha 1.0"
)

# ─────── Submit Loop ───────
declare -a temp_files=()

# Cleanup temp files on exit (even if script fails or hope is slow)
cleanup() {
    if [ ${#temp_files[@]} -gt 0 ]; then
        echo "--- Cleaning up temp HOPE files ---" | tee -a "$LOG_FILE"
        rm -f "${temp_files[@]}"
    fi
}
trap cleanup EXIT

for exp_str in "${experiments[@]}"; do
    IFS='|' read -r MODE EXTRA_ARGS <<< "$exp_str"

    OUTPUT_DIR="${BASE_OUTPUT}/${MODE}"
    HOPE_NAME="qwen3_cpt_mg_${MODE}"

    echo "--- Submitting: ${HOPE_NAME} ---" | tee -a "$LOG_FILE"
    echo "  Mode:   ${MODE}" | tee -a "$LOG_FILE"
    echo "  Output: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"

    mkdir -p "${OUTPUT_DIR}"

    WORKER_CMD="export HF_DATA_SOURCES='${HF_DATA_SOURCES}' && bash ${BRIDGE_SCRIPT_DIR}/run_cpt.sh ${MODE} ${OUTPUT_DIR} __multisource__ \"${EXTRA_ARGS}\""

    # Generate temp HOPE file with robust worker.script replacement
    TEMP_HOPE="${HOPE_NAME}.hope"
    python3 -c "
import re, sys
with open('${HOPE_TEMPLATE}') as f:
    content = f.read()
cmd = '''${WORKER_CMD}'''
# Robust replacement: match 'worker.script = <anything>' on the whole line
content = re.sub(
    r'^(worker\.script\s*=\s*).*$',
    r'\g<1>' + cmd.strip().replace('\\\\', '\\\\\\\\'),
    content,
    count=1,
    flags=re.MULTILINE,
)
with open('${TEMP_HOPE}', 'w') as f:
    f.write(content)
"
    temp_files+=("${TEMP_HOPE}")

    echo "  Generated: ${TEMP_HOPE}" | tee -a "$LOG_FILE"

    # ──── Submit (sequential to avoid race with temp file reads) ────
    hope run "${TEMP_HOPE}" \
        -Dhope.resource.experiment=syn_fmg_h800 \
        -Dmlp.sche.priority=P0
    echo "  Submitted ✓" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
echo "  ALL 6 EXPERIMENTS SUBMITTED — 500B TOKENS (NO REPETITION)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Hyperparameters:" | tee -a "$LOG_FILE"
echo "    LR=1e-4, min_LR=1e-5, warmup=2000" | tee -a "$LOG_FILE"
echo "    SEQ=4096, GBS=512 (2.10M tok/step)" | tee -a "$LOG_FILE"
echo "    ITERS=238K (~500B tokens total)" | tee -a "$LOG_FILE"
echo "    4 nodes × 8 H800, TP=4, DP=8" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Data mixture (no repetition, DCLM downsampled):" | tee -a "$LOG_FILE"
echo "    dclm_dedup:       84.26% web  (421.3B of 750B)" | tee -a "$LOG_FILE"
echo "    nemotron_cc_math: 12.32% math ( 61.6B — all)" | tee -a "$LOG_FILE"
echo "    stack_edu_python:  3.42% code ( 17.1B — all)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Est. wall time: 30-45 days @ 32 GPUs" | tee -a "$LOG_FILE"
echo "  → Scale to 64 GPUs (8 nodes) for 15-22 days" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  Monitor: hope status" | tee -a "$LOG_FILE"
echo "  Outputs: ${BASE_OUTPUT}/" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
