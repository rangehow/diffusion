#!/bin/bash
# ==============================================================================
# Evaluate MDLM checkpoint using PrefixLM-style (tail-biased) MC collator
#
# Purpose: Test whether MDLM's uniform-masking-trained model can score well
#          when the noise is concentrated at the tail (suffix) of continuation
#           explain PrefixLM's poor downstream performance.
#
# Key difference from normal MDLM eval:
#   --diffusion_type prefixlm   (routes to PrefixLMMCCollator)
#   instead of
#   --diffusion_type mdlm       (routes to MDLMMCCollator)
#
# The PrefixLMMCCollator masks the LAST t fraction of continuation tokens
# (contiguous suffix), matching PrefixLM training noise distribution.
# ==============================================================================

set -e

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

HOPE_TEMPLATE="hope/h800_m17_1node.hope"
MAIN_SCRIPT_PATH="script/sh/eval_sh.sh"

# --- Model: use the MDLM main experiment checkpoint ---
MODEL_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/mdlm_main_exp/checkpoint-77335"

# --- Eval config ---
# KEY: diffusion_type = prefixlm → uses PrefixLMMCCollator (tail/suffix masking)
BATCH_MODEL_TYPE="discrete_diffusion"
DIFFUSION_TYPE="prefixlm"
DIFFUSION_EVAL_MODE="mc"
MC_NUM=32
BATCH_SIZE=16
LIMIT=0
OUTPUT_DIR="./evaluation_results_mdlm_as_prefixlm"

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

echo "=== MDLM evaluated as PrefixLM (tail masking) ==="
echo "Queue:      h800_m17_public"
echo "Nodes:      1 (8 GPUs)"
echo "Model:      ${MODEL_DIR}"
echo "Model type: ${BATCH_MODEL_TYPE}"
echo "Diff type:  ${DIFFUSION_TYPE}  ← tail/suffix masking"
echo "Eval mode:  ${DIFFUSION_EVAL_MODE} (mc_num=${MC_NUM})"
echo "Tasks:      ${TASKS_JSON}"
echo "Batch:      ${BATCH_SIZE}"
echo "Output:     ${OUTPUT_DIR}"
echo "=================================================="

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

TEMP_HOPE="eval_mdlm_as_prefixlm.hope"
ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

echo "Submitting HOPE job..."
hope run "${TEMP_HOPE}" -Dhope.resource.experiment=syn_fmg_h800 -Dmlp.sche.priority=P0
HOPE_EXIT=$?

rm -f "${TEMP_HOPE}"

if [ ${HOPE_EXIT} -eq 0 ]; then
    echo "MDLM-as-PrefixLM eval job submitted successfully."
else
    echo "ERROR: HOPE submission failed with exit code ${HOPE_EXIT}"
    exit ${HOPE_EXIT}
fi
