#!/bin/bash
# ==============================================================================
# Parallel Evaluation: Submit one HOPE job per benchmark task
#
# Instead of running all 8 tasks sequentially on 1 node, this script submits
# 8 separate HOPE jobs — one task per node (8 GPUs each). All tasks run in
# parallel, reducing wall-clock time from ~8× to ~1× of a single task.
#
# Usage:
#   bash sh_eval_parallel.sh                              # Use defaults
#   bash sh_eval_parallel.sh /path/to/checkpoint          # Specify checkpoint
#   bash sh_eval_parallel.sh /path/to/ckpt prefixlm      # Specify checkpoint + diffusion_type
#
# Arguments (all optional, with sensible defaults):
#   $1 = MODEL_DIR         (default: MDLM main exp checkpoint)
#   $2 = DIFFUSION_TYPE    (default: mdlm; options: mdlm, prefixlm, causal, bd3lm)
#   $3 = OUTPUT_DIR        (default: auto-generated from model name + diffusion_type)
#   $4 = HOPE_TEMPLATE     (default: hope/h800_m17_1node.hope)
#   $5 = MC_NUM            (default: 32)
#   $6 = BATCH_SIZE        (default: 16)
#
# Each HOPE job gets 1 node (8 GPUs) and evaluates 1 task.
# Results are written to the same output directory, merged by the log file naming.
# ==============================================================================

set -e

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

# --- Arguments with defaults ---
MODEL_DIR="${1:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/mdlm_main_exp/checkpoint-77335}"
DIFFUSION_TYPE="${2:-mdlm}"
OUTPUT_DIR="${3:-}"
HOPE_TEMPLATE="${4:-hope/h800_m17_1node.hope}"
MC_NUM="${5:-32}"
BATCH_SIZE="${6:-16}"

MAIN_SCRIPT_PATH="script/sh/eval_sh.sh"
BATCH_MODEL_TYPE="discrete_diffusion"
DIFFUSION_EVAL_MODE="mc"
LIMIT=0

# --- Auto-generate output dir if not specified ---
if [ -z "${OUTPUT_DIR}" ]; then
    MODEL_NAME=$(basename "$(dirname "${MODEL_DIR}")")
    CKPT_NAME=$(basename "${MODEL_DIR}")
    OUTPUT_DIR="./evaluation_results_parallel/${MODEL_NAME}_${DIFFUSION_TYPE}"
fi

# --- Task definitions: {"task_name": num_fewshot} ---
# Each task becomes one HOPE job
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
    echo "ERROR: HOPE template not found: ${HOPE_TEMPLATE}"
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

echo "============================================================"
echo "  Parallel Evaluation — 1 HOPE job per task"
echo "============================================================"
echo "Model:         ${MODEL_DIR}"
echo "Diff type:     ${DIFFUSION_TYPE}"
echo "Eval mode:     ${DIFFUSION_EVAL_MODE} (mc_num=${MC_NUM})"
echo "Batch size:    ${BATCH_SIZE}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "HOPE template: ${HOPE_TEMPLATE}"
echo "Tasks:         ${!TASKS[*]}"
echo "Total jobs:    ${#TASKS[@]}"
echo "============================================================"
echo ""

# --- Submit one HOPE job per task ---
SUBMITTED=0
FAILED=0

for TASK_NAME in "${!TASKS[@]}"; do
    TASK_SHOT="${TASKS[$TASK_NAME]}"
    
    # Each task gets its own single-task JSON
    TASK_JSON="{\"${TASK_NAME}\":${TASK_SHOT}}"
    
    echo "--- Submitting: ${TASK_NAME} (${TASK_SHOT}-shot) ---"
    
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
    
    TEMP_HOPE="eval_parallel_${TASK_NAME}.hope"
    ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
    sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"
    
    set +e
    hope run "${TEMP_HOPE}" -Dhope.resource.experiment=syn_fmg_h800 -Dmlp.sche.priority=P0
    HOPE_EXIT=$?
    set -e
    
    rm -f "${TEMP_HOPE}"
    
    if [ ${HOPE_EXIT} -eq 0 ]; then
        echo "  ✓ ${TASK_NAME} submitted"
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "  ✗ ${TASK_NAME} FAILED (exit code ${HOPE_EXIT})"
        FAILED=$((FAILED + 1))
    fi
    
    # Small delay between submissions to avoid overwhelming the scheduler
    sleep 2
done

echo ""
echo "============================================================"
echo "  Submission Summary"
echo "============================================================"
echo "  Submitted: ${SUBMITTED}/${#TASKS[@]}"
echo "  Failed:    ${FAILED}/${#TASKS[@]}"
echo "  Output:    ${OUTPUT_DIR}"
echo ""
echo "  All ${SUBMITTED} tasks will run in PARALLEL (1 node each)."
echo "  Results will be written to the same output directory."
echo "============================================================"

if [ ${FAILED} -gt 0 ]; then
    exit 1
fi
