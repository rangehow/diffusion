#!/bin/bash
# =========================================================================
#  PrefixLM 1B 正式训练 — m17 队列加速版
#  32 nodes = 256 GPUs, batch_size=16, grad_accum=2
#  有效 batch = 256 × 16 × 2 = 8192 seqs/step (是原来 4096 的 2 倍)
#  => 步数减半 ~38668 steps, 训练时间几乎减半
#
#  加速策略:
#    1. bs 8→16  (显存只占一半, 放心翻倍)
#    2. ga 4→2   (保持有效 batch 从 4096 翻倍到 8192)
#    3. 256卡比 128卡 多一倍并行度
#    4. save_steps 加大到 2000 减少 IO 开销
#    5. dataloader_num_workers 提到 16 避免 CPU 瓶颈
# =========================================================================

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

HOPE_TEMPLATE="hope/h800_m17_32node.hope"
MAIN_SCRIPT_PATH="script/sh/sh_train.sh"

# --- 训练参数 ---
OUTPUT_DIR="diffusion/model_output/prefixlm_main_exp_m17"
MLM_SCHEDULE_TYPE=""
MLM_PROB_START=1
MLM_PROB_END=0.0001
BATCH_SIZE=16
# 256 GPUs × bs=16 × ga=2 = 8192 seqs/step
GRAD_ACCUM_STEPS=2
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

echo "=== PrefixLM 1B Full Training (m17 加速版) ==="
echo "Queue: h800_m17_public"
echo "Nodes: 32 (256 GPUs)"
echo "bs=16, ga=2 => eff_batch = 16 * 2 * 256 = 8192 seqs/step"
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

TEMP_HOPE="prefixlm_full_train_m17.hope"
ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

echo "Submitting HOPE job to m17 queue..."
hope run "${TEMP_HOPE}" -Dhope.resource.experiment=syn_fmg_h800 -Dmlp.sche.priority=P0
HOPE_EXIT=$?

rm -f "${TEMP_HOPE}"

if [ ${HOPE_EXIT} -eq 0 ]; then
    echo "PrefixLM full training job (m17) submitted successfully."
else
    echo "ERROR: HOPE submission failed with exit code ${HOPE_EXIT}"
    exit ${HOPE_EXIT}
fi
