#!/bin/bash
# =========================================================================
#  URGENT: Retrain MDLM only — 4 nodes × 8 GPUs = 32 GPUs per task
#
#  EOS masking fix requires retraining. This script submits ONLY MDLM jobs
#  on 4 nodes for maximum speed.
#
#  Usage:
#    cd /path/to/diffusion/hope
#    bash script/sh/submit_mdlm_retrain.sh                       # FMG (default)
#    bash script/sh/submit_mdlm_retrain.sh --queue m17           # M17
#    bash script/sh/submit_mdlm_retrain.sh --queue m17 --dry-run # preview
#    bash script/sh/submit_mdlm_retrain.sh --nodes 2             # fallback to 2 nodes
#    bash script/sh/submit_mdlm_retrain.sh --task sudoku          # single task
# =========================================================================

set -euo pipefail

# ---- Defaults ----
DRY_RUN=false
FORCE=false
TASK_FILTER=""
QUEUE="fmg"
NODES=4
MODEL_SIZE="110m"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --force)   FORCE=true; shift ;;
        --task)    TASK_FILTER="$2"; shift 2 ;;
        --queue)   QUEUE="$2"; shift 2 ;;
        --nodes)   NODES="$2"; shift 2 ;;
        --size)    MODEL_SIZE="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Queue-specific settings ----
if [ "${QUEUE}" = "fmg" ]; then
    EXPERIMENT_FLAG="fmg_h800_ci"
    HOPE_TEMPLATE_NAME="reasoning_fmg_${NODES}node.hope"
elif [ "${QUEUE}" = "m17" ]; then
    EXPERIMENT_FLAG="syn_fmg_h800"
    HOPE_TEMPLATE_NAME="reasoning_m17_${NODES}node.hope"
else
    echo "ERROR: --queue must be 'fmg' or 'm17'"; exit 1
fi

TOTAL_GPUS=$((NODES * 8))

echo "============================================="
echo "  MDLM RETRAIN → ${QUEUE^^} Queue"
echo "  ${NODES} nodes × 8 GPUs = ${TOTAL_GPUS} GPUs per task"
echo "  Experiment: ${EXPERIMENT_FLAG}"
echo "  Model: ${MODEL_SIZE}"
echo "============================================="
[ "${DRY_RUN}" = true ] && echo "  *** DRY RUN ***"
[ "${FORCE}" = true ]   && echo "  *** FORCE ***"
[ -n "${TASK_FILTER}" ] && echo "  Task filter: ${TASK_FILTER}"
echo "============================================="
echo ""

# ---- Paths ----
DIFFUSION_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion"
OUTPUT_BASE="${DIFFUSION_DIR}/model_output"
HOPE_DIR="${DIFFUSION_DIR}/hope"

cd "${HOPE_DIR}"

HOPE_TEMPLATE="hope/${HOPE_TEMPLATE_NAME}"
TRAIN_SCRIPT="script/sh/train_reasoning.sh"

# Verify
if [ ! -f "${HOPE_TEMPLATE}" ]; then
    echo "ERROR: HOPE template not found: ${HOPE_TEMPLATE}"
    echo "Available templates:"
    ls -1 hope/reasoning_*.hope 2>/dev/null || echo "  (none)"
    exit 1
fi

JOB_COUNT=0
SKIPPED=0

submit_mdlm() {
    local TASK=$1
    local EPOCHS=$2
    local BATCH_SIZE=$3
    local MAX_LENGTH=$4
    local LR=$5
    local MODE="reasoning_mdlm"

    if [ -n "${TASK_FILTER}" ] && [ "${TASK}" != "${TASK_FILTER}" ]; then
        return
    fi

    local JOB_NAME="${TASK}_${MODE}_${MODEL_SIZE}"
    local JOB_OUTPUT_DIR="${OUTPUT_BASE}/reasoning_${JOB_NAME}"

    # Skip if completed
    if [ "${FORCE}" = false ] && [ -f "${JOB_OUTPUT_DIR}/trainer_state.json" ]; then
        echo "[SKIP] ${JOB_NAME}  (already completed)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    # Move old checkpoint aside if it exists (from EOS-never-masked training)
    if [ -d "${JOB_OUTPUT_DIR}" ] && [ ! -f "${JOB_OUTPUT_DIR}/trainer_state.json" ]; then
        local OLD_DIR="${JOB_OUTPUT_DIR}_eos_v2"
        if [ ! -d "${OLD_DIR}" ]; then
            echo "  Moving old incomplete dir → ${OLD_DIR}"
            mv "${JOB_OUTPUT_DIR}" "${OLD_DIR}" 2>/dev/null || true
        fi
    fi

    # With 32 GPUs: per_device_bs * 32 * grad_accum = 1024
    # BS=32 → 32*32=1024, grad_accum=1 ✓
    # For Sudoku with MAX_LENGTH=165: BS=16 → 16*32=512, grad_accum=2 → eff=1024
    echo "----------------------------------------------"
    echo "[SUBMIT] ${JOB_NAME}  (${NODES} nodes, ${TOTAL_GPUS} GPUs)"
    echo "  epochs=${EPOCHS}  bs=${BATCH_SIZE}  maxlen=${MAX_LENGTH}  lr=${LR}"

    local WORKER_CMD="bash ${TRAIN_SCRIPT} ${TASK} ${MODE} ${EPOCHS} ${BATCH_SIZE} ${MAX_LENGTH} ${LR} ${MODEL_SIZE}"

    local TEMP_HOPE="reasoning_${JOB_NAME}.hope"
    local ESCAPED_CMD
    ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
    sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

    if [ "${DRY_RUN}" = true ]; then
        echo "  [DRY RUN] hope run ${TEMP_HOPE}"
        echo "  cmd: ${WORKER_CMD}"
        rm -f "${TEMP_HOPE}"
    else
        hope run "${TEMP_HOPE}" -Dhope.resource.experiment=${EXPERIMENT_FLAG} -Dmlp.sche.priority=P0
        local EXIT_CODE=$?
        rm -f "${TEMP_HOPE}"
        if [ ${EXIT_CODE} -eq 0 ]; then
            echo "  ✓ Submitted"
        else
            echo "  ✗ FAILED (exit ${EXIT_CODE})"
        fi
    fi
    JOB_COUNT=$((JOB_COUNT + 1))
    echo ""
}

# ---- Submit MDLM tasks ----
# 110M model on 141GB H800: use large per-GPU batch to maximize utilization
# With 32 GPUs and BS=256/gpu: eff_batch=8192 (8x Ye et al.'s 1024)
# LR scaled by sqrt(8)≈2.83: 3e-4 * 2.83 = 8.5e-4
# CD:     36K total steps (enough for 600 epochs), ~25-35GB VRAM/gpu
# Sudoku: BS=128/gpu → eff_batch=4096, ~30-40GB VRAM/gpu, 14K steps
#
# This fills ~25% of 141GB VRAM while keeping training dynamics healthy

submit_mdlm cd3     600  256  37   8.5e-4
submit_mdlm cd4     600  256  64   8.5e-4
submit_mdlm cd5     600  256  74   8.5e-4
submit_mdlm sudoku  300  128  165  8.5e-4

echo "============================================="
echo "  Summary"
echo "============================================="
echo "  Submitted : ${JOB_COUNT} MDLM jobs"
echo "  Skipped   : ${SKIPPED}"
echo "  GPUs/job  : ${TOTAL_GPUS} (${NODES} nodes × 8)"
echo "  Queue     : ${QUEUE^^}"
echo "============================================="
