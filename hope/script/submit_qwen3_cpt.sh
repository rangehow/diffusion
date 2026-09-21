#!/bin/bash
# ============================================================
# Submit all 5 Qwen3-8B CPT experiments to HOPE m17 queue
#
# Experiments:
#   1. arm           — Pure ARM baseline
#   2. card_naive    — CARD V1 (scattered mask + DAUM)
#   3. card_v3       — CARD V3 (topological reorder + logical pos IDs)
#   4. adapter_only  — Suffix mask + adapter (freeze backbone)
#   5. adapter_full  — Suffix mask + adapter (train everything)
#
# Usage:
#   cd /path/to/diffusion/hope/script
#   bash submit_qwen3_cpt.sh
#
# Before submitting:
#   1. Fill in DATASET_NAME below with your actual dataset
#   2. Adjust NODES, BATCH_SIZE, GRAD_ACCUM as needed
#   3. Verify the HOPE_TEMPLATE path
# ============================================================

set -e

# ─────── Global Config ───────
DATASET_NAME="qwen3_cpt_mix"
BASE_OUTPUT="diffusion/model_output/qwen3_cpt"
HOPE_TEMPLATE="../hope/qwen3_cpt_m17.hope"
WORKER_SCRIPT="qwen3_cpt.sh"
NODES=4                                    # 4 nodes × 8 GPUs = 32 GPUs
MAX_LENGTH=2048
LOG_FILE="./qwen3_cpt_submission.log"

# Per-experiment hyperparameters
# Qwen3-8B: ~8B params, 2048 seq len, bf16
# 4 nodes × 8 GPUs = 32 GPUs
# Global batch = 32 × batch_size × grad_accum
# Target: ~512K tokens/step → batch=2, accum=4 → 32×2×4×2048 = 524K tokens/step
BATCH_SIZE=2
GRAD_ACCUM=4
LR="2e-5"

echo "--- Qwen3-8B CPT Submission Script ---" | tee "$LOG_FILE"
echo "Dataset: ${DATASET_NAME}" | tee -a "$LOG_FILE"
echo "Output base: ${BASE_OUTPUT}" | tee -a "$LOG_FILE"
echo "Nodes: ${NODES}" | tee -a "$LOG_FILE"
echo "Global batch: $((NODES * 8 * BATCH_SIZE * GRAD_ACCUM * MAX_LENGTH)) tokens/step" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ─────── Experiment Definitions ───────
# Format: "mode|output_suffix|extra_args"
experiments=(
    "arm|arm|"
    "card_naive|card_naive|--mlm_start_prob 0.99999 --mlm_end_prob 0.00001 --tail_bias_factor 1.5 --use_daum True"
    "card_v3|card_v3|--mlm_start_prob 0.99999 --mlm_end_prob 0.00001 --tail_bias_factor 1.5 --use_daum False"
    "adapter_only|adapter_only|--adapter_rank 64 --schedule_alpha 1.0 --learning_rate 3e-4"
    "adapter_full|adapter_full|--adapter_rank 64 --schedule_alpha 1.0"
)

# ─────── Submit Loop ───────
declare -a temp_files=()

for exp_str in "${experiments[@]}"; do
    IFS='|' read -r MODE SUFFIX EXTRA_ARGS <<< "$exp_str"
    
    OUTPUT_DIR="${BASE_OUTPUT}/${SUFFIX}"
    HOPE_NAME="qwen3_cpt_${SUFFIX}"
    
    echo "--- Submitting: ${HOPE_NAME} ---" | tee -a "$LOG_FILE"
    echo "  Mode: ${MODE}" | tee -a "$LOG_FILE"
    echo "  Output: ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
    echo "  Extra: ${EXTRA_ARGS}" | tee -a "$LOG_FILE"
    
    # Adapter-only uses higher LR for adapter params
    if [ "$MODE" = "adapter_only" ]; then
        CURR_LR="3e-4"
    else
        CURR_LR="$LR"
    fi
    
    # Build worker command
    WORKER_CMD="bash ${WORKER_SCRIPT} ${MODE} ${OUTPUT_DIR} ${DATASET_NAME} ${MAX_LENGTH} ${BATCH_SIZE} ${GRAD_ACCUM} ${CURR_LR} \"${EXTRA_ARGS}\""
    
    # Generate temp HOPE file
    TEMP_HOPE="${HOPE_NAME}.hope"
    ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
    
    sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" \
        "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"
    
    # Also set the correct number of nodes
    sed -i "s#^workers = .*#workers = ${NODES}#" "${TEMP_HOPE}"
    
    temp_files+=("${TEMP_HOPE}")
    
    echo "  Generated: ${TEMP_HOPE}" | tee -a "$LOG_FILE"
    echo "  Command: ${WORKER_CMD}" | tee -a "$LOG_FILE"
    
    # ──── Submit ────
    hope run "${TEMP_HOPE}" &
    echo "  Submitted: hope run ${TEMP_HOPE}" | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
done

# ─────── Wait and Cleanup ───────
echo "--- Waiting for all submissions ---" | tee -a "$LOG_FILE"
wait

echo "--- Cleaning up temp HOPE files ---" | tee -a "$LOG_FILE"
rm -f "${temp_files[@]}"

echo "" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
echo "  ALL 5 EXPERIMENTS SUBMITTED" | tee -a "$LOG_FILE"
echo "  Monitor with: hope status" | tee -a "$LOG_FILE"
echo "  Outputs in: ${BASE_OUTPUT}/" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
echo ""
echo "All experiments submitted."
