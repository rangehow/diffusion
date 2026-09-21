#!/bin/bash
# =========================================================================
#  Eval runner for MDM (Ye et al. 2024 "Beyond Autoregression") experiments.
#  Runs all 4 tasks (cd3, cd4, cd5, sudoku) on a single GPU each.
#
#  Usage:
#    cd /path/to/diffusion
#    CUDA_VISIBLE_DEVICES=0,1,2,3 bash hope/script/sh/run_eval_mdm_reasoning.sh
#
#  Or via HOPE:
#    MODEL_SIZE=110m bash hope/script/sh/run_eval_mdm_reasoning.sh
# =========================================================================
set -eo pipefail

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04"
DIFFUSION_DIR="${BASE_DIR}/diffusion"

source "${BASE_DIR}/.bashrc"
conda activate base

EVAL_OUTPUT_DIR="${DIFFUSION_DIR}/reasoning_tasks/eval_output"
MODEL_SIZE="${MODEL_SIZE:-110m}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
# Ye et al. use stochastic TopK decoding
DECODING="${DECODING_STRATEGY:-stochastic0.5-linear}"

mkdir -p "${EVAL_OUTPUT_DIR}"
cd "${DIFFUSION_DIR}"

echo "============================================="
echo "  MDM Reasoning Eval | model_size=${MODEL_SIZE}"
echo "  Decoding: ${DECODING}"
echo "  Inference batch_size=${BATCH_SIZE}"
echo "  Time: $(date)"
echo "============================================="

# Format: TASK:MAX_NEW_TOKENS:NUM_STEPS:MAX_SEQ_LEN
ALL_TASKS=(
    "cd3:24:20:37"
    "cd4:32:20:64"
    "cd5:54:20:74"
    "sudoku:81:20:165"
)

PIDS=()
TASK_NAMES=()
GPU_ID=0

for ENTRY in "${ALL_TASKS[@]}"; do
    IFS=':' read -r TASK MAX_NEW NUM_STEPS MAX_SEQ_LEN <<< "${ENTRY}"
    
    MODE="reasoning_mdm"
    CKPT_MODE="reasoning_mdm"
    
    RESULT_FILE="${EVAL_OUTPUT_DIR}/${TASK}_${MODE}_${MODEL_SIZE}.json"
    
    if [ -f "${RESULT_FILE}" ]; then
        echo "[GPU ${GPU_ID}] SKIP ${TASK} × ${MODE} (result exists)"
        GPU_ID=$((GPU_ID + 1))
        continue
    fi
    
    MODEL_DIR="${DIFFUSION_DIR}/model_output/reasoning_${TASK}_${CKPT_MODE}_${MODEL_SIZE}"
    CKPT=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [ -z "${CKPT}" ]; then
        if [ -f "${MODEL_DIR}/config.json" ]; then
            CKPT="${MODEL_DIR}"
        else
            echo "[GPU ${GPU_ID}] NO MODEL for ${TASK} × ${MODE}, skipping"
            GPU_ID=$((GPU_ID + 1))
            continue
        fi
    fi
    
    LOG_FILE="${EVAL_OUTPUT_DIR}/${TASK}_${MODE}_${MODEL_SIZE}.log"
    echo ""
    echo "[GPU ${GPU_ID}] Launching: ${TASK} × ${MODE}"
    echo "    checkpoint: ${CKPT}"
    echo "    max_new=${MAX_NEW} steps=${NUM_STEPS} batch_size=${BATCH_SIZE} max_seq_len=${MAX_SEQ_LEN}"
    echo "    decoding: ${DECODING}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m reasoning_tasks.evaluate \
        --model_path "${CKPT}" \
        --task "${TASK}" \
        --mode "${MODE}" \
        --data_dir "${DIFFUSION_DIR}/reasoning_tasks/data" \
        --output_file "${RESULT_FILE}" \
        --max_new_tokens "${MAX_NEW}" \
        --num_steps "${NUM_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --temperature 0.0 \
        --device cuda \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --decoding_strategy "${DECODING}" \
        2>&1 | tee "${LOG_FILE}" &
    
    PIDS+=($!)
    TASK_NAMES+=("${TASK}")
    GPU_ID=$((GPU_ID + 1))
done

echo ""
echo "[INFO] ${#PIDS[@]} tasks launched, waiting..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    NAME="${TASK_NAMES[$i]}"
    if ! wait ${PID}; then
        echo "[FAILED] ${NAME} (PID ${PID})"
        FAILED=$((FAILED + 1))
    else
        echo "[DONE] ${NAME} (PID ${PID})"
    fi
done

echo ""
echo "============================================="
echo "  MDM Results (model_size=${MODEL_SIZE})"
echo "============================================="
printf "  %-8s %-10s %s\n" "TASK" "ACCURACY" "THROUGHPUT"
printf "  %-8s %-10s %s\n" "--------" "----------" "----------"
for ENTRY in "${ALL_TASKS[@]}"; do
    IFS=':' read -r TASK _ _ _ <<< "${ENTRY}"
    RESULT_FILE="${EVAL_OUTPUT_DIR}/${TASK}_reasoning_mdm_${MODEL_SIZE}.json"
    if [ -f "${RESULT_FILE}" ]; then
        STATS=$(python3 -c "
import json
d=json.load(open('${RESULT_FILE}'))
acc=f'{d[\"accuracy\"]*100:.1f}%'
tp=d.get('throughput_tok_per_sec', 0)
tp_str=f'{tp:,.0f} tok/s' if tp else 'N/A'
ca=d.get('cell_accuracy')
ca_str=f'  cell={ca*100:.1f}%' if ca is not None else ''
print(f'{acc}|{tp_str}|{ca_str}')
" 2>/dev/null || echo "ERR|ERR|")
        IFS='|' read -r ACC TP EXTRA <<< "${STATS}"
        printf "  %-8s %-10s %s%s\n" "${TASK}" "${ACC}" "${TP}" "${EXTRA}"
    else
        printf "  %-8s MISSING\n" "${TASK}"
    fi
done
echo "============================================="
echo "Failed: ${FAILED} | Time: $(date)"
exit ${FAILED}
