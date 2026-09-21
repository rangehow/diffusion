#!/bin/bash
# =========================================================================
#  Submit reasoning task experiments to M17 queue (h800_m17_public)
#
#  Follows Ye et al. (2024) "Beyond Autoregression" setting.
#  Tasks:  CD3, CD4, CD5, Sudoku
#  Modes:  CARD, AR, MDLM
#  Sizes:  110m (default), 400m (~320M)
#
#  Features:
#    - Skip-if-done: detects trainer_state.json and skips completed jobs
#    - Resubmit safe: just run again to pick up failed tasks
#    - --force to ignore existing results
#    - --dry-run to preview without submitting
#
#  Usage:
#    cd /path/to/diffusion/hope
#    bash script/sh/submit_reasoning_m17.sh                                   # 110m only
#    bash script/sh/submit_reasoning_m17.sh --size 400m                       # 400m only
#    bash script/sh/submit_reasoning_m17.sh --size both                       # 110m + 400m
#    bash script/sh/submit_reasoning_m17.sh --dry-run                         # preview
#    bash script/sh/submit_reasoning_m17.sh --force                           # resubmit all
#    bash script/sh/submit_reasoning_m17.sh --task cd3 --mode reasoning_card  # single job
# =========================================================================

set -euo pipefail

# ---- Defaults ----
DRY_RUN=false
SIZE_FILTER="110m"
FORCE=false
TASK_FILTER=""
MODE_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --size)    SIZE_FILTER="$2"; shift 2 ;;
        --force)   FORCE=true; shift ;;
        --task)    TASK_FILTER="$2"; shift 2 ;;
        --mode)    MODE_FILTER="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================="
echo "  Reasoning Tasks → M17 Queue (h800_m17_public)"
echo "  Experiment: syn_fmg_h800"
echo "============================================="
[ "${DRY_RUN}" = true ]  && echo "  *** DRY RUN MODE ***"
[ "${FORCE}" = true ]    && echo "  *** FORCE MODE ***"
echo "  Size filter: ${SIZE_FILTER}"
[ -n "${TASK_FILTER}" ]  && echo "  Task filter: ${TASK_FILTER}"
[ -n "${MODE_FILTER}" ]  && echo "  Mode filter: ${MODE_FILTER}"
echo "============================================="
echo ""

# ---- Paths ----
DIFFUSION_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion"
OUTPUT_BASE="${DIFFUSION_DIR}/model_output"
HOPE_DIR="${DIFFUSION_DIR}/hope"

cd "${HOPE_DIR}"

HOPE_TEMPLATE="hope/reasoning_m17_1node.hope"
TRAIN_SCRIPT="script/sh/train_reasoning.sh"

# ---- Verify template + script exist ----
for f in "${HOPE_TEMPLATE}" "${TRAIN_SCRIPT}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: File not found: ${f}"
        exit 1
    fi
done

# ---- Counters ----
JOB_COUNT=0
SKIPPED=0
FAILED=0

# ---- Submit one job ----
submit_job() {
    local TASK=$1
    local MODE=$2
    local EPOCHS=$3
    local BATCH_SIZE=$4
    local MAX_LENGTH=$5
    local LR=$6
    local MODEL_SIZE=$7

    # Apply filters
    if [ -n "${TASK_FILTER}" ] && [ "${TASK}" != "${TASK_FILTER}" ]; then
        return
    fi
    if [ -n "${MODE_FILTER}" ] && [ "${MODE}" != "${MODE_FILTER}" ]; then
        return
    fi

    local JOB_NAME="${TASK}_${MODE}_${MODEL_SIZE}"
    local JOB_OUTPUT_DIR="${OUTPUT_BASE}/reasoning_${JOB_NAME}"

    # Skip if already completed
    if [ "${FORCE}" = false ] && [ -f "${JOB_OUTPUT_DIR}/trainer_state.json" ]; then
        echo "[SKIP] ${JOB_NAME}  (trainer_state.json exists)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    # Info
    echo "----------------------------------------------"
    echo "[SUBMIT] ${JOB_NAME}"
    echo "  epochs=${EPOCHS}  bs=${BATCH_SIZE}  maxlen=${MAX_LENGTH}  lr=${LR}"
    if [ -d "${JOB_OUTPUT_DIR}" ]; then
        echo "  (dir exists but incomplete — resubmitting)"
    fi

    local WORKER_CMD="bash ${TRAIN_SCRIPT} ${TASK} ${MODE} ${EPOCHS} ${BATCH_SIZE} ${MAX_LENGTH} ${LR} ${MODEL_SIZE}"

    # Create temp hope file
    local TEMP_HOPE="reasoning_${JOB_NAME}.hope"
    local ESCAPED_CMD
    ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
    sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

    if [ "${DRY_RUN}" = true ]; then
        echo "  [DRY RUN] hope run ${TEMP_HOPE}"
        echo "  cmd: ${WORKER_CMD}"
        rm -f "${TEMP_HOPE}"
    else
        hope run "${TEMP_HOPE}" -Dhope.resource.experiment=syn_fmg_h800 -Dmlp.sche.priority=P0
        local EXIT_CODE=$?
        rm -f "${TEMP_HOPE}"

        if [ ${EXIT_CODE} -eq 0 ]; then
            echo "  ✓ Submitted"
        else
            echo "  ✗ FAILED (exit ${EXIT_CODE})"
            FAILED=$((FAILED + 1))
        fi
    fi
    JOB_COUNT=$((JOB_COUNT + 1))
    echo ""
}

# ---- Determine sizes ----
SIZES=()
case "${SIZE_FILTER}" in
    110m)  SIZES=("110m") ;;
    400m)  SIZES=("400m") ;;
    both)  SIZES=("110m" "400m") ;;
    *)     echo "ERROR: --size must be 110m, 400m, or both"; exit 1 ;;
esac

# ---- Submit all jobs ----
for MODEL_SIZE in "${SIZES[@]}"; do
    echo ""
    echo "===== Model Size: ${MODEL_SIZE} ====="
    echo ""

    # Batch size adjustment for 400m (larger model → smaller BS)
    if [ "${MODEL_SIZE}" = "400m" ]; then
        CD_BS=64
        SUDOKU_BS=32
    else
        CD_BS=128
        SUDOKU_BS=64
    fi

    # Countdown: Ye et al. settings
    submit_job cd3 reasoning_card  600 ${CD_BS}     37  3e-4 ${MODEL_SIZE}
    submit_job cd3 reasoning_ar    600 ${CD_BS}     37  3e-4 ${MODEL_SIZE}
    submit_job cd3 reasoning_mdlm  600 ${CD_BS}     37  3e-4 ${MODEL_SIZE}
    submit_job cd3 reasoning_mdm   600 ${CD_BS}     37  3e-4 ${MODEL_SIZE}

    submit_job cd4 reasoning_card  600 ${CD_BS}     64  3e-4 ${MODEL_SIZE}
    submit_job cd4 reasoning_ar    600 ${CD_BS}     64  3e-4 ${MODEL_SIZE}
    submit_job cd4 reasoning_mdlm  600 ${CD_BS}     64  3e-4 ${MODEL_SIZE}
    submit_job cd4 reasoning_mdm   600 ${CD_BS}     64  3e-4 ${MODEL_SIZE}

    submit_job cd5 reasoning_card  600 ${CD_BS}     74  3e-4 ${MODEL_SIZE}
    submit_job cd5 reasoning_ar    600 ${CD_BS}     74  3e-4 ${MODEL_SIZE}
    submit_job cd5 reasoning_mdlm  600 ${CD_BS}     74  3e-4 ${MODEL_SIZE}
    submit_job cd5 reasoning_mdm   600 ${CD_BS}     74  3e-4 ${MODEL_SIZE}

    # Sudoku: cutoff_len=165 (BOS+81puzzle+SEP+81solution+EOS), 300 epochs
    submit_job sudoku reasoning_card  300 ${SUDOKU_BS}  165 3e-4 ${MODEL_SIZE}
    submit_job sudoku reasoning_ar    300 ${SUDOKU_BS}  165 3e-4 ${MODEL_SIZE}
    submit_job sudoku reasoning_mdlm  300 ${SUDOKU_BS}  165 3e-4 ${MODEL_SIZE}
    submit_job sudoku reasoning_mdm   300 ${SUDOKU_BS}  165 3e-4 ${MODEL_SIZE}
done

# ---- Summary ----
echo "============================================="
echo "  Summary"
echo "============================================="
echo "  Submitted : ${JOB_COUNT}"
echo "  Skipped   : ${SKIPPED} (already completed)"
[ ${FAILED} -gt 0 ] && echo "  Failed    : ${FAILED}"
echo ""
echo "  Queue     : h800_m17_public"
echo "  Experiment: syn_fmg_h800"
echo "  GPUs/job  : 8 (1 node)"
echo ""
if [ ${SKIPPED} -gt 0 ]; then
    echo "  Skipped jobs have trainer_state.json in output dir."
    echo "  Use --force to resubmit them."
    echo ""
fi
echo "  Output dirs: ${OUTPUT_BASE}/reasoning_*/"
echo ""
echo "  Tip: Run again to resubmit only failed/incomplete tasks."
echo "============================================="
