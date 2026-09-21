#!/bin/bash
# =========================================================================
#  PrefixLM 1B 正式训练 — fmg 队列 512卡版 (bug-fixed collator)
#  64 nodes = 512 GPUs, batch_size=16, grad_accum=1
#  有效 batch = 512 × 16 × 1 = 8192 seqs/step (与 256 卡版一致)
#  => 步数不变 ~38668 steps, 训练时间再减半
# =========================================================================

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

HOPE_TEMPLATE="hope/h800_fmg_64node.hope"
MAIN_SCRIPT_PATH="script/sh/sh_train.sh"

# --- 训练参数 ---
OUTPUT_DIR="diffusion/model_output/prefixlm_main_exp_fmg_512"
MLM_SCHEDULE_TYPE=""
MLM_PROB_START=1
MLM_PROB_END=0.0001
BATCH_SIZE=16
# 512 GPUs × bs=16 × ga=1 = 8192 seqs/step (与 256 卡版一致)
GRAD_ACCUM_STEPS=1
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

echo "=== PrefixLM 1B Full Training (fmg 512卡版) ==="
echo "Queue: h800_fmg_exp"
echo "Nodes: 64 (512 GPUs)"
echo "bs=16, ga=1 => eff_batch = 16 * 1 * 512 = 8192 seqs/step"
echo "Approx steps: ~38668 (1 epoch)"
echo "Config: llada_1b.json (same backbone as MDLM)"
echo "Data: sh_filtered_finefineweb"
echo "Output: ${OUTPUT_DIR}"
echo "================================================"

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

TEMP_HOPE="prefixlm_full_train_fmg_512.hope"
ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

echo "Submitting HOPE job to fmg queue..."
hope run "${TEMP_HOPE}" -Dhope.resource.experiment=syn_fmg_h800 -Dmlp.sche.priority=P0
HOPE_EXIT=$?

rm -f "${TEMP_HOPE}"

if [ ${HOPE_EXIT} -eq 0 ]; then
    echo "PrefixLM full training job (fmg 512-GPU) submitted successfully."
else
    echo "ERROR: HOPE submission failed with exit code ${HOPE_EXIT}"
    exit ${HOPE_EXIT}
fi
