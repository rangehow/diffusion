#!/bin/bash
# ==============================================================================
# Parallel Evaluation on FMG Queue: Submit one HOPE job per benchmark task
#
# Each of the 8 tasks runs on its own 1-node (8 GPU) FMG job in parallel.
# Wall-clock time ≈ slowest single task (not 8× sequential).
#
# Usage:
#   # 1) Evaluate MDLM checkpoint with MDLM-style masking (baseline)
#   bash sh_eval_parallel_fmg.sh
#
#   # 2) Evaluate MDLM checkpoint with PrefixLM-style tail masking
#   bash sh_eval_parallel_fmg.sh \
#       /path/to/mdlm/checkpoint \
#       prefixlm
#
#   # 3) Full control
#   bash sh_eval_parallel_fmg.sh \
#       /path/to/checkpoint \
#       prefixlm \
#       ./my_output_dir \
#       32 \
#       16
#
# Arguments (all optional):
#   $1 = MODEL_DIR       (default: MDLM main exp checkpoint-77335)
#   $2 = DIFFUSION_TYPE  (default: mdlm; options: mdlm, prefixlm)
#   $3 = OUTPUT_DIR      (default: auto-generated)
#   $4 = MC_NUM          (default: 32)
#   $5 = BATCH_SIZE      (default: 16)
# ==============================================================================

set -e

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

# --- Arguments with defaults ---
MODEL_DIR="${1:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/mdlm_main_exp/checkpoint-77335}"
DIFFUSION_TYPE="${2:-mdlm}"
OUTPUT_DIR="${3:-}"
MC_NUM="${4:-32}"
BATCH_SIZE="${5:-16}"

# --- FMG-specific settings ---
HOPE_TEMPLATE="hope/sh_fmg_1node.hope"
HOPE_EXPERIMENT="fmg_h800_ci"
HOPE_PRIORITY="P0"

MAIN_SCRIPT_PATH="script/sh/eval_sh.sh"
BATCH_MODEL_TYPE="discrete_diffusion"
DIFFUSION_EVAL_MODE="mc"
LIMIT=0

# --- Auto-generate output dir if not specified ---
if [ -z "${OUTPUT_DIR}" ]; then
    MODEL_NAME=$(basename "$(dirname "${MODEL_DIR}")")
    CKPT_NAME=$(basename "${MODEL_DIR}")
    # If MODEL_DIR itself is a model dir (not checkpoint-*), use its basename
    if [[ "${CKPT_NAME}" != checkpoint-* ]]; then
        MODEL_NAME="${CKPT_NAME}"
        CKPT_NAME=""
    fi
    OUTPUT_DIR="./evaluation_results_parallel/${MODEL_NAME}_${DIFFUSION_TYPE}"
fi

# --- All 8 benchmark tasks matching the paper's Table 1 ---
declare -A TASKS
TASKS=(
    ["hellaswag"]=3
    ["mmlu_redux_corrected"]=5
    ["arc_easy"]=25
    ["arc_challenge"]=25
    ["piqa"]=0
    ["winogrande"]=5
    ["commonsense_qa"]=7
    ["sciq"]=0
)

# --- Validation ---
if [ ! -f "${HOPE_TEMPLATE}" ]; then
    echo "ERROR: FMG HOPE template not found: ${HOPE_TEMPLATE}"
    echo "       Expected at: $(pwd)/${HOPE_TEMPLATE}"
    exit 1
fi
if [ ! -f "${MAIN_SCRIPT_PATH}" ]; then
    echo "ERROR: Eval script not found: ${MAIN_SCRIPT_PATH}"
    exit 1
fi
if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: Model checkpoint not found: ${MODEL_DIR}"
    exit 1
fi

# --- Pre-flight: fix checkpoints & clear stale HF cache ---
echo "[Pre-flight] Fixing checkpoint compatibility (transformers >= 5.3)..."
PROJ_ROOT="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion"
python3 "${PROJ_ROOT}/fix_checkpoints_transformers53.py" 2>&1 | tail -3
echo "[Pre-flight] Clearing stale __pycache__ in HF cache..."
find /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/.cache/modules/transformers_modules/ \
    -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo ""

echo "============================================================"
echo "  FMG Parallel Evaluation — 1 job per task"
echo "============================================================"
echo "Queue:      h800_fmg_exp"
echo "Experiment: ${HOPE_EXPERIMENT}"
echo "Nodes:      1 × ${#TASKS[@]} jobs (8 GPUs each)"
echo ""
echo "Model:      ${MODEL_DIR}"
echo "Diff type:  ${DIFFUSION_TYPE}"
echo "Eval mode:  ${DIFFUSION_EVAL_MODE} (mc_num=${MC_NUM})"
echo "Batch size: ${BATCH_SIZE}"
echo "Output:     ${OUTPUT_DIR}"
echo ""
echo "Tasks:"
for TASK_NAME in "${!TASKS[@]}"; do
    echo "  - ${TASK_NAME} (${TASKS[$TASK_NAME]}-shot)"
done
echo "============================================================"
echo ""

# --- Submit one HOPE job per task ---
SUBMITTED=0
FAILED=0

for TASK_NAME in "${!TASKS[@]}"; do
    TASK_SHOT="${TASKS[$TASK_NAME]}"
    TASK_JSON="{\"${TASK_NAME}\":${TASK_SHOT}}"

    echo "--- Submitting: ${TASK_NAME} (${TASK_SHOT}-shot) ---"

    # Build the worker command (runs inside the HOPE container)
    WORKER_CMD="bash ${MAIN_SCRIPT_PATH} \
        \"${OUTPUT_DIR}\" \
        \"${MODEL_DIR}\" \
        \"${BATCH_MODEL_TYPE}\" \
        \"${BATCH_SIZE}\" \
        \"${LIMIT}\" \
        \"${DIFFUSION_EVAL_MODE}\" \
        '${TASK_JSON}' \
        \"${DIFFUSION_TYPE}\" \
        \"${MC_NUM}\""

    # Generate per-task .hope file from FMG template
    TEMP_HOPE="eval_fmg_${TASK_NAME}.hope"
    ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
    sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

    # Submit to FMG queue
    set +e
    hope run "${TEMP_HOPE}" \
        -Dhope.resource.experiment=${HOPE_EXPERIMENT} \
        -Dmlp.sche.priority=${HOPE_PRIORITY}
    HOPE_EXIT=$?
    set -e

    rm -f "${TEMP_HOPE}"

    if [ ${HOPE_EXIT} -eq 0 ]; then
        echo "  ✓ ${TASK_NAME} submitted to FMG"
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "  ✗ ${TASK_NAME} FAILED (exit code ${HOPE_EXIT})"
        FAILED=$((FAILED + 1))
    fi

    # Brief pause between submissions
    sleep 2
done

echo ""
echo "============================================================"
echo "  FMG Submission Summary"
echo "============================================================"
echo "  Submitted: ${SUBMITTED}/${#TASKS[@]}"
echo "  Failed:    ${FAILED}/${#TASKS[@]}"
echo "  Output:    ${OUTPUT_DIR}"
echo ""
echo "  All ${SUBMITTED} tasks run in PARALLEL on FMG queue."
echo "  Each task gets 1 node (8 GPUs)."
echo "  Results merge into the same output directory."
echo "============================================================"

if [ ${FAILED} -gt 0 ]; then
    exit 1
fi
