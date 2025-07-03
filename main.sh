#!/bin/bash
export NUMEXPR_MAX_THREADS=1000

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/.bashrc

mamba activate sglang
echo $HF_HOME

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04
export WANDB_DISABLED=true
# 确保脚本在遇到错误时退出
set -e

# --- 配置你的路径和参数 ---

# 1. 路径设置
# 将这里的路径替换为你的实际路径
MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/answerdotai/ModernBERT-base/main"
# DATASET_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/EleutherAI/fineweb-edu-dedup-10b/main"
DATASET_NAME="ultra_fineweb"


# 2. 训练超参数
EPOCHS=1
LEARNING_RATE=5e-4
BATCH_SIZE=2 # per_device_train_batch_size
GRAD_ACCUM_STEPS=8

OUTPUT_DIR="diffusion/model_output/debug_linear_llada" # 训练输出目录

MLM_SCHEDULE_TYPE=random
MLM_PROB_START=1
MLM_PROB_END=0
RANDOM_PROB=0
MASK_PROB=1
# CONFIG_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_large.json
CONFIG_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llama_400M.json
# CONFIG_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llada_large.json
use_all_tokens_for_loss=false
MODE=llama


CUDA_VISIBLE_DEVICES=0 accelerate launch -m diffusion.main \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_path "${CONFIG_PATH}" \
  --num_train_epochs ${EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --dataloader_num_workers 8 \
  --warmup_ratio 0.01 \
  --mlm_start_prob "${MLM_PROB_START}" \
  --mlm_end_prob "${MLM_PROB_END}" \
  --random_probability "${RANDOM_PROB}" \
  --mask_probability "${MASK_PROB}" \
  --logging_steps 10 \
  --save_total_limit 2 \
  --seed 42 \
  --max_length 2500 \
  --mlm_schedule_type "${MLM_SCHEDULE_TYPE}" \
  --mode "${MODE}" \
  --use_all_tokens_for_loss ${use_all_tokens_for_loss} \
  --bf16

echo "训练脚本执行完毕。"