#!/bin/bash
# 速度基准测试训练脚本 — 1 node (8 GPUs), 500 steps
# 参数与 sh_train.sh 完全一致，仅额外接收 MAX_STEPS

# --- 从命令行参数中读取配置 ---
OUTPUT_DIR=$1
MLM_SCHEDULE_TYPE=$2
MLM_PROB_START=$3
MLM_PROB_END=$4
BATCH_SIZE=$5
GRAD_ACCUM_STEPS=$6
CONFIG_PATH=$7
MODE=${8}
DATASET_NAME=${9}
MAX_LENGTH=${10}
EPOCHS=${11}
tail_bias_factor=${12}
use_daum=${13}
MAX_STEPS=${14}

echo "--- 速度基准测试配置 ---"
echo "输出目录: ${OUTPUT_DIR}"
echo "模式: ${MODE}"
echo "Batch Size (per device): ${BATCH_SIZE}"
echo "梯度累积步数: ${GRAD_ACCUM_STEPS}"
echo "最大长度: ${MAX_LENGTH}"
echo "最大步数: ${MAX_STEPS}"
echo "Config: ${CONFIG_PATH}"
echo "------------------------"

set -e
export NUMEXPR_MAX_THREADS=1000

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/.bashrc
conda activate base
echo "HF_HOME is set to: $HF_HOME"

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04
export WANDB_DISABLED=true

MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/ModernBERT-base"
LEARNING_RATE=2e-4

launch=$(python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope/script/acc.py)
echo $launch

# Detect correct network interface (route-based, avoids RDMA-only NIC like eth9)
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope/script/detect_nccl_ifname.sh
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,NET,GRAPH,ENV,DYNDBG
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=20
export NCCL_IB_RETRY_CNT=15
export NCCL_IB_QPS_PER_CONNECTION=8

# BD3LM 需要 pad_to_max_length（与原始 main_exp 一致）
# 其他模型不 pad（MDLM 原始也是 pad_to_max_length=False）
PAD_FLAG=""
if [ "${MODE}" = "bd3lm" ]; then
    PAD_FLAG="--pad_to_max_length"
fi

${launch} -m diffusion.pretrain.main \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_path "${CONFIG_PATH}" \
  --num_train_epochs ${EPOCHS} \
  --max_steps ${MAX_STEPS} \
  --learning_rate ${LEARNING_RATE} \
  --tail_bias_factor ${tail_bias_factor} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --dataloader_num_workers 8 \
  --warmup_ratio 0.01 \
  --mlm_start_prob "${MLM_PROB_START}" \
  --mlm_end_prob "${MLM_PROB_END}" \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 499 \
  --save_total_limit 1 \
  --seed 42 \
  --max_length "${MAX_LENGTH}" \
  --mode "${MODE}" \
  --use_daum "${use_daum}" \
  --lr_scheduler_kwargs "{'min_lr_rate': 0.01}" \
  ${PAD_FLAG} \
  --bf16

echo "基准测试完成: ${MODE}"
