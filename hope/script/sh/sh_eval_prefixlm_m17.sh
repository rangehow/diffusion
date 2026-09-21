#!/bin/bash
# ==============================================================================
# PrefixLM Evaluation — M17 queue, 1 node (8 GPUs)
# Evaluates the final checkpoint on all 8 downstream tasks + PPL
# ==============================================================================

set -e

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

HOPE_TEMPLATE="hope/h800_m17_1node.hope"
MAIN_SCRIPT_PATH="script/sh/eval_sh.sh"

# --- Model ---
MODEL_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/prefixlm_main_exp_m17/checkpoint-38668"

# --- Eval config ---
# PrefixLM uses LLaDA backbone (bidirectional) → same eval as MDLM/LLaDA
BATCH_MODEL_TYPE="discrete_diffusion"
DIFFUSION_TYPE="mdlm"
DIFFUSION_EVAL_MODE="mc"
MC_NUM=32
BATCH_SIZE=16
LIMIT=0
OUTPUT_DIR="./evaluation_results"

# All 8 tasks matching the paper's Table 1
TASKS_JSON='{"hellaswag":3,"mmlu_redux_corrected":5,"arc_easy":25,"arc_challenge":25,"piqa":0,"winogrande":5,"commonsense_qa":7,"sciq":0}'

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

echo "=== PrefixLM Evaluation (M17) ==="
echo "Queue:  h800_m17_public"
echo "Nodes:  1 (8 GPUs)"
echo "Model:  ${MODEL_DIR}"
echo "Type:   ${BATCH_MODEL_TYPE} / ${DIFFUSION_TYPE}"
echo "Mode:   ${DIFFUSION_EVAL_MODE} (mc_num=${MC_NUM})"
echo "Tasks:  ${TASKS_JSON}"
echo "Batch:  ${BATCH_SIZE}"
echo "=================================="

WORKER_CMD="bash ${MAIN_SCRIPT_PATH} \
    \"${OUTPUT_DIR}\" \
    \"${MODEL_DIR}\" \
    \"${BATCH_MODEL_TYPE}\" \
    \"${BATCH_SIZE}\" \
    \"${LIMIT}\" \
    \"${DIFFUSION_EVAL_MODE}\" \
    '${TASKS_JSON}' \
    \"${DIFFUSION_TYPE}\" \
    \"${MC_NUM}\""

TEMP_HOPE="eval_prefixlm_m17.hope"
ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

echo "Submitting HOPE job..."
hope run "${TEMP_HOPE}" -Dhope.resource.experiment=syn_fmg_h800 -Dmlp.sche.priority=P0
HOPE_EXIT=$?

rm -f "${TEMP_HOPE}"

if [ ${HOPE_EXIT} -eq 0 ]; then
    echo "PrefixLM eval job submitted successfully."
else
    echo "ERROR: HOPE submission failed with exit code ${HOPE_EXIT}"
    exit ${HOPE_EXIT}
fi
