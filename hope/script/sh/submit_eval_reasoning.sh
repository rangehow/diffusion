#!/bin/bash
# =========================================================================
#  Submit reasoning eval to FMG queue as TWO 1-node jobs (8 GPUs each).
#    Job A (batch 0): 8 tasks (cd3×4 + cd4×4)
#    Job B (batch 1): 8 tasks (cd5×4 + sudoku×4)
#
#  4 modes per task: reasoning_card, reasoning_ar, reasoning_mdlm, reasoning_card_ar
#  card_ar = CARD model generating token-by-token (like AR)
#
#  Usage:
#    cd /path/to/diffusion/hope
#    bash script/sh/submit_eval_reasoning.sh                  # 110m
#    bash script/sh/submit_eval_reasoning.sh --size 400m
#    bash script/sh/submit_eval_reasoning.sh --dry-run
# =========================================================================
set -euo pipefail

DRY_RUN=false
MODEL_SIZE="110m"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --size)    MODEL_SIZE="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

HOPE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope"
DIFFUSION_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion"
HOPE_TEMPLATE="${HOPE_DIR}/hope/reasoning_fmg_1node.hope"
EVAL_SCRIPT="script/sh/run_eval_reasoning_batch.sh"
EVAL_OUTPUT_DIR="${DIFFUSION_DIR}/reasoning_tasks/eval_output"

# --- Check files ---
if [ ! -f "${HOPE_TEMPLATE}" ]; then
    echo "ERROR: HOPE template not found: ${HOPE_TEMPLATE}"
    exit 1
fi
if [ ! -f "${HOPE_DIR}/${EVAL_SCRIPT}" ]; then
    echo "ERROR: Eval script not found: ${HOPE_DIR}/${EVAL_SCRIPT}"
    exit 1
fi

echo "============================================="
echo "  Reasoning Eval → FMG Queue"
echo "  2 jobs × 1 node × 8 GPUs"
echo "  4 modes: CARD, AR, MDLM, CARD_AR (token-by-token)"
echo "  Model size: ${MODEL_SIZE}"
echo "  Inference batch_size: 256"
echo "============================================="

cd "${HOPE_DIR}"

SUBMITTED=0
SKIPPED=0

for BATCH_ID in 0 1; do
    if [ "${BATCH_ID}" -eq 0 ]; then
        DESC="batch0 (8 tasks: cd3×4 + cd4×4)"
        TASKS_TO_CHECK=(
            "cd3_reasoning_card" "cd3_reasoning_ar" "cd3_reasoning_mdlm" "cd3_reasoning_card_ar"
            "cd4_reasoning_card" "cd4_reasoning_ar" "cd4_reasoning_mdlm" "cd4_reasoning_card_ar"
        )
    else
        DESC="batch1 (8 tasks: cd5×4 + sudoku×4)"
        TASKS_TO_CHECK=(
            "cd5_reasoning_card" "cd5_reasoning_ar" "cd5_reasoning_mdlm" "cd5_reasoning_card_ar"
            "sudoku_reasoning_card" "sudoku_reasoning_ar" "sudoku_reasoning_mdlm" "sudoku_reasoning_card_ar"
        )
    fi

    JOB_NAME="eval_reasoning_b${BATCH_ID}_${MODEL_SIZE}"

    # --- Check if ALL tasks in this batch are already done ---
    ALL_DONE=true
    for T in "${TASKS_TO_CHECK[@]}"; do
        if [ ! -f "${EVAL_OUTPUT_DIR}/${T}_${MODEL_SIZE}.json" ]; then
            ALL_DONE=false
            break
        fi
    done

    if [ "${ALL_DONE}" = true ]; then
        echo "[SKIP] ${JOB_NAME} — all results exist"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # --- Build worker command ---
    WORKER_CMD="cd ${DIFFUSION_DIR} && MODEL_SIZE=${MODEL_SIZE} BATCH_ID=${BATCH_ID} EVAL_BATCH_SIZE=256 bash hope/${EVAL_SCRIPT}"

    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY-RUN] ${JOB_NAME} — ${DESC}"
        echo "    script: ${WORKER_CMD}"
        echo ""
    else
        echo "Submitting: ${JOB_NAME} — ${DESC}"

        TEMP_HOPE="${JOB_NAME}.hope"
        ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
        sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

        hope run "${TEMP_HOPE}" -Dhope.resource.experiment=fmg_h800_ci -Dmlp.sche.priority=P0
        HOPE_EXIT=$?

        rm -f "${TEMP_HOPE}"

        if [ ${HOPE_EXIT} -eq 0 ]; then
            echo "  → submitted"
        else
            echo "  WARNING: hope run exited with ${HOPE_EXIT}"
        fi
        SUBMITTED=$((SUBMITTED + 1))
        echo ""
    fi
done

echo "============================================="
echo "Summary:"
echo "  Submitted: ${SUBMITTED} jobs"
echo "  Skipped:   ${SKIPPED} (all results exist)"
echo "  Logs will be at: ${EVAL_OUTPUT_DIR}/*.log"
echo "============================================="
