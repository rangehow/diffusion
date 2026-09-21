#!/bin/bash
# =========================================================================
#  PrefixLM 1B 正式训练 (16 nodes = 128 GPUs, 77335 steps, 1 epoch)
#  使用 LLaDA backbone (双向注意力) + PrefixLM collator
#  完全对齐 MDLM/CARD main_exp 的训练量
#
#  复用 sh_train.sh + h800_sh_2node.hope (workers=16 = 128 GPUs)
# =========================================================================

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

HOPE_TEMPLATE="hope/h800_sh_2node.hope"
MAIN_SCRIPT_PATH="script/sh/sh_train.sh"

# --- 参数完全对齐 MDLM main_exp ---
OUTPUT_DIR="diffusion/model_output/prefixlm_main_exp"
MLM_SCHEDULE_TYPE=""
MLM_PROB_START=1
MLM_PROB_END=0.0001
BATCH_SIZE=8
# 128 GPUs × bs=8 × ga=4 = 4096 seqs/step (与 MDLM/CARD 一致)
GRAD_ACCUM_STEPS=4
CONFIG_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config/llada_1b.json"
MODE="prefixlm"
DATASET_NAME="sh_filtered_finefineweb"
MAX_LENGTH=2048
EPOCHS=1
TAIL_BIAS_FACTOR=1.5
USE_DAUM=True

# --- 检查文件 ---
if [ ! -f "${HOPE_TEMPLATE}" ]; then
    echo "ERROR: HOPE template not found: ${HOPE_TEMPLATE}"
    exit 1
fi
if [ ! -f "${MAIN_SCRIPT_PATH}" ]; then
    echo "ERROR: Training script not found: ${MAIN_SCRIPT_PATH}"
    exit 1
fi

echo "=== PrefixLM 1B Full Training ==="
echo "Nodes: 16 (128 GPUs)"
echo "bs=8, ga=4 => eff_batch = 8 * 4 * 128 = 4096 seqs/step"
echo "Steps: 77335 (1 epoch)"
echo "Config: llada_1b.json (same backbone as MDLM)"
echo "Data: sh_filtered_finefineweb"
echo "Output: ${OUTPUT_DIR}"
echo "================================="

WORKER_CMD="bash ${MAIN_SCRIPT_PATH} \
    \"${OUTPUT_DIR}\" \
    \"${MLM_SCHEDULE_TYPE}\" \
    \"${MLM_PROB_START}\" \
    \"${MLM_PROB_END}\" \
    \"${BATCH_SIZE}\" \
    \"${GRAD_ACCUM_STEPS}\" \
    \"${CONFIG_PATH}\" \
    \"${MODE}\" \
    \"${DATASET_NAME}\" \
    \"${MAX_LENGTH}\" \
    \"${EPOCHS}\" \
    \"${TAIL_BIAS_FACTOR}\" \
    \"${USE_DAUM}\""

TEMP_HOPE="prefixlm_full_train.hope"
ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

echo "Submitting HOPE job..."
hope run "${TEMP_HOPE}" -Dhope.resource.experiment=fmg_h800_ci -Dmlp.sche.priority=P0
HOPE_EXIT=$?

rm -f "${TEMP_HOPE}"

if [ ${HOPE_EXIT} -eq 0 ]; then
    echo "PrefixLM full training job submitted successfully."
else
    echo "ERROR: HOPE submission failed with exit code ${HOPE_EXIT}"
    exit ${HOPE_EXIT}
fi
