#!/bin/bash
# =========================================================================
#  Eval runner: runs a batch of eval tasks in parallel, one per GPU.
#  Called by HOPE with env vars: MODEL_SIZE, BATCH_ID
#  
#  16 tasks total (4 modes × 4 tasks):
#    reasoning_card, reasoning_ar, reasoning_mdlm, reasoning_card_ar
#  
#  BATCH_ID=0 → tasks 0–7  (8 GPUs)
#  BATCH_ID=1 → tasks 8–15 (8 GPUs)
# =========================================================================
set -eo pipefail

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04"
DIFFUSION_DIR="${BASE_DIR}/diffusion"
# Tokenizer is built-in (ReasoningCharTokenizer) — no path needed
TOKENIZER_PATH=""

# ---- Activate conda environment ----
source "${BASE_DIR}/.bashrc"
conda activate base

EVAL_OUTPUT_DIR="${DIFFUSION_DIR}/reasoning_tasks/eval_output"
MODEL_SIZE="${MODEL_SIZE:-110m}"
BATCH_ID="${BATCH_ID:-0}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"

mkdir -p "${EVAL_OUTPUT_DIR}"
cd "${DIFFUSION_DIR}"

echo "============================================="
echo "  Eval Batch ${BATCH_ID} | model_size=${MODEL_SIZE}"
echo "  Inference batch_size=${BATCH_SIZE}"
echo "  GPU count: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  Time: $(date)"
echo "============================================="

# ---- All 16 tasks ----
# Format: TASK:MODE:MAX_NEW_TOKENS:NUM_STEPS:BLOCK_SIZE:MODEL_MODE_FOR_CKPT:MAX_SEQ_LEN
# MODEL_MODE_FOR_CKPT: which training mode's checkpoint to load (card_ar reuses card's checkpoint)
# MAX_SEQ_LEN: training max_length — CRITICAL for MDLM to match training attention distribution
#   CD3=37, CD4=64, CD5=74, Sudoku=165
ALL_TASKS=(
    # Batch 0: cd3 (4 modes) + cd4 (4 modes) = 8
    # max_new_tokens from Ye et al.: CD3=24, CD4=32, CD5=54, Sudoku=82
    "cd3:reasoning_card:24:16:24:reasoning_card:37"
    "cd3:reasoning_ar:24:0:0:reasoning_ar:37"
    "cd3:reasoning_mdlm:24:20:0:reasoning_mdlm:37"
    "cd3:reasoning_card_ar:24:1:1:reasoning_card:37"
    "cd4:reasoning_card:32:16:32:reasoning_card:64"
    "cd4:reasoning_ar:32:0:0:reasoning_ar:64"
    "cd4:reasoning_mdlm:32:20:0:reasoning_mdlm:64"
    "cd4:reasoning_card_ar:32:1:1:reasoning_card:64"
    # Batch 1: cd5 (4 modes) + sudoku (4 modes) = 8
    "cd5:reasoning_card:54:16:32:reasoning_card:74"
    "cd5:reasoning_ar:54:0:0:reasoning_ar:74"
    "cd5:reasoning_mdlm:54:20:0:reasoning_mdlm:74"
    "cd5:reasoning_card_ar:54:1:1:reasoning_card:74"
    "sudoku:reasoning_card:81:32:81:reasoning_card:165"
    "sudoku:reasoning_ar:81:0:0:reasoning_ar:165"
    "sudoku:reasoning_mdlm:81:20:0:reasoning_mdlm:165"
    "sudoku:reasoning_card_ar:81:1:1:reasoning_card:165"
    # Batch 2: MDM (Ye et al. 2024) — 4 tasks using stochastic TopK decoding
    "cd3:reasoning_mdm:24:20:0:reasoning_mdm:37"
    "cd4:reasoning_mdm:32:20:0:reasoning_mdm:64"
    "cd5:reasoning_mdm:54:20:0:reasoning_mdm:74"
    "sudoku:reasoning_mdm:81:20:0:reasoning_mdm:165"
)

# ---- Split: batch 0 → [0..7], batch 1 → [8..15], batch 2 → [16..19] ----
if [ "${BATCH_ID}" -eq 0 ]; then
    START=0; END=7
    echo "[Batch 0] Running 8 tasks (cd3×4 + cd4×4), 8 GPUs"
elif [ "${BATCH_ID}" -eq 1 ]; then
    START=8; END=15
    echo "[Batch 1] Running 8 tasks (cd5×4 + sudoku×4), 8 GPUs"
elif [ "${BATCH_ID}" -eq 2 ]; then
    START=16; END=19
    echo "[Batch 2] Running 4 tasks (MDM: cd3+cd4+cd5+sudoku), 4 GPUs"
else
    echo "Unknown BATCH_ID=${BATCH_ID}"; exit 1
fi

# ---- Launch tasks in parallel ----
PIDS=()
TASK_NAMES=()
GPU_ID=0

for IDX in $(seq ${START} ${END}); do
    ENTRY="${ALL_TASKS[$IDX]}"
    IFS=':' read -r TASK MODE MAX_NEW NUM_STEPS BLOCK_SIZE CKPT_MODE MAX_SEQ_LEN <<< "${ENTRY}"

    RESULT_FILE="${EVAL_OUTPUT_DIR}/${TASK}_${MODE}_${MODEL_SIZE}.json"

    # Skip if result already exists
    if [ -f "${RESULT_FILE}" ]; then
        echo "[GPU ${GPU_ID}] SKIP ${TASK} × ${MODE} (result exists)"
        GPU_ID=$((GPU_ID + 1))
        continue
    fi

    # Find model checkpoint (card_ar uses card's checkpoint)
    MODEL_DIR="${DIFFUSION_DIR}/model_output/reasoning_${TASK}_${CKPT_MODE}_${MODEL_SIZE}"
    CKPT=""
    CKPT=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [ -z "${CKPT}" ]; then
        if [ -f "${MODEL_DIR}/config.json" ]; then
            CKPT="${MODEL_DIR}"
        else
            echo "[GPU ${GPU_ID}] NO MODEL for ${TASK} × ${MODE} (looked in ${MODEL_DIR}), skipping"
            GPU_ID=$((GPU_ID + 1))
            continue
        fi
    fi

    LOG_FILE="${EVAL_OUTPUT_DIR}/${TASK}_${MODE}_${MODEL_SIZE}.log"
    echo ""
    echo "[GPU ${GPU_ID}] Launching: ${TASK} × ${MODE}"
    echo "    checkpoint: ${CKPT}"
    echo "    max_new=${MAX_NEW} steps=${NUM_STEPS} block=${BLOCK_SIZE} batch_size=${BATCH_SIZE} max_seq_len=${MAX_SEQ_LEN}"

    # Build command — add --max_seq_len for MDLM/MDM (critical for matching training distribution)
    EXTRA_ARGS=""
    if [[ "${MODE}" == *"mdlm"* || "${MODE}" == *"mdm"* ]] && [ -n "${MAX_SEQ_LEN}" ]; then
        EXTRA_ARGS="--max_seq_len ${MAX_SEQ_LEN}"
    fi
    # MDM uses stochastic TopK decoding (Ye et al.)
    if [[ "${MODE}" == *"mdm"* ]]; then
        EXTRA_ARGS="${EXTRA_ARGS} --decoding_strategy stochastic0.5-linear"
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m reasoning_tasks.evaluate \
        --model_path "${CKPT}" \
        --task "${TASK}" \
        --mode "${MODE}" \
        --data_dir "${DIFFUSION_DIR}/reasoning_tasks/data" \
        --output_file "${RESULT_FILE}" \
        --max_new_tokens "${MAX_NEW}" \
        --num_steps "${NUM_STEPS}" \
        --block_size "${BLOCK_SIZE}" \
        --batch_size "${BATCH_SIZE}" \
        --temperature 0.0 \
        --device cuda \
        ${EXTRA_ARGS} \
        2>&1 | tee "${LOG_FILE}" &

    PIDS+=($!)
    TASK_NAMES+=("${TASK}:${MODE}")
    GPU_ID=$((GPU_ID + 1))
done

echo ""
echo "[INFO] ${#PIDS[@]} tasks launched, waiting for completion..."
echo ""

# ---- Wait and track ----
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

# ---- Summary ----
echo ""
echo "============================================="
echo "  Batch ${BATCH_ID} Results (model_size=${MODEL_SIZE})"
echo "============================================="
printf "  %-8s %-25s %-10s %s\n" "TASK" "MODE" "ACCURACY" "THROUGHPUT"
printf "  %-8s %-25s %-10s %s\n" "--------" "-------------------------" "----------" "----------"
for IDX in $(seq ${START} ${END}); do
    ENTRY="${ALL_TASKS[$IDX]}"
    IFS=':' read -r TASK MODE _ _ _ _ <<< "${ENTRY}"
    RESULT_FILE="${EVAL_OUTPUT_DIR}/${TASK}_${MODE}_${MODEL_SIZE}.json"
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
        printf "  %-8s %-25s %-10s %s%s\n" "${TASK}" "${MODE}" "${ACC}" "${TP}" "${EXTRA}"
    else
        LOG_FILE="${EVAL_OUTPUT_DIR}/${TASK}_${MODE}_${MODEL_SIZE}.log"
        if [ -f "${LOG_FILE}" ]; then
            LAST_LINE=$(tail -1 "${LOG_FILE}" 2>/dev/null || echo "no log")
            printf "  %-8s %-25s FAILED (%s)\n" "${TASK}" "${MODE}" "${LAST_LINE:0:50}"
        else
            printf "  %-8s %-25s MISSING\n" "${TASK}" "${MODE}"
        fi
    fi
done
echo "============================================="
echo "Failed: ${FAILED} | Time: $(date)"

exit ${FAILED}
