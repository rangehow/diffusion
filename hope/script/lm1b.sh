#!/bin/bash
# lm1b.sh

# --- 1. 接收参数 ---
OUTPUT_DIR=$1
MODE=$2
CONFIG_FILE=$3
DATASET_NAME=$4
EPOCHS=$5
BATCH_SIZE=$6
MAX_STEPS=$7
GRADIENT_ACCUMULATION_STEPS=$8
EMA_DECAY=$9
LR_SCHEDULER_KWARGS=${10}
LR_SCHEDULER_TYPE=${11}
# (修改 1/2) 新增: 接收第12和13个参数
WARMUP_STEPS=${12}
WARMUP_RATIOS=${13}

# --- 2. 固定环境与路径设置 ---
# 确保脚本在遇到错误时立即退出
set -e

# 设置环境变量和激活Conda环境
export NUMEXPR_MAX_THREADS=1000
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/.bashrc
mamba activate sglang
echo "Conda environment 'sglang' activated."
echo "HF_HOME is set to: $HF_HOME"

# 进入工作目录
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04
export WANDB_DISABLED=true

# 模型配置文件基础路径 (固定)
BASE_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion"
# 拼接完整的模型配置文件路径
CONFIG_PATH="${BASE_PATH}/model_config/${CONFIG_FILE}"
echo "完整的模型配置文件路径: ${CONFIG_PATH}"


# --- 3. 执行训练命令 ---
launch=$(python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/hope/acc.py)
echo $launch

# (修改 2/2) 将硬编码的参数替换为从命令行传入的变量
${launch} -m diffusion.pretrain.main \
  --model_name_or_path "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/hldy/model/google-bert/bert-base-uncased/main" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_path "${CONFIG_PATH}" \
  --num_train_epochs ${EPOCHS} \
  --max_steps ${MAX_STEPS}\
  --learning_rate "3e-4" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --lr_scheduler_kwargs "${LR_SCHEDULER_KWARGS}" \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --mlm_start_prob 1 \
  --mlm_end_prob 0.0001 \
  --mode "${MODE}" \
  --max_length 128 \
  --per_device_eval_batch_size 512 \
  --validation_dataset_name lm1b_test \
  --eval_steps 5000 \
  --logging_steps 100 \
  --save_steps 5000 \
  --save_total_limit 2 \
  --dataloader_num_workers 8 \
  --warmup_ratio ${WARMUP_RATIOS} \
  --warmup_steps ${WARMUP_STEPS} \
  --seed 42 \
  --ema_decay ${EMA_DECAY} \
  --bf16 \
  --pad_to_max_length

echo "训练任务完成。"