#!/bin/bash
export NUMEXPR_MAX_THREADS=1000

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/.bashrc

mamba activate sglang
echo $HF_HOME

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04
export WANDB_DISABLED=true
set -e

# --- 配置你的路径和参数 ---

# 1. 路径设置
# 将这里的路径替换为你的实际路径
MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/hldy/model/google-bert/bert-base-uncased/main"

DATASET_NAME="lm1b_train"

# 2. 训练超参数
EPOCHS=2
LEARNING_RATE=3e-4
BATCH_SIZE=2 # per_device_train_batch_size, packing模式下通常为1
GRAD_ACCUM_STEPS=1


MLM_SCHEDULE_TYPE=random
MLM_PROB_START=1
MLM_PROB_END=0

# CONFIG_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_large.json
# CONFIG_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llama_400M.json
# 模型配置文件基础路径
BASE_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion"

# # 模型配置文件名
# OUTPUT_DIR="diffusion/model_output/debug_llada"
# CONFIG_FILE="llada_110m.json"
# MODE=mdlm


# OUTPUT_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/debug_modernbert
# CONFIG_FILE="modernbert_110m.json"
# MODE=niu



# OUTPUT_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/debug_arm
# CONFIG_FILE="llama_110m.json"
# MODE=llama


OUTPUT_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_output/debug_bd3lm
CONFIG_FILE="bd3lm_110m.json"
MODE=bd3lm

CONFIG_PATH="${BASE_PATH}/model_config/${CONFIG_FILE}"
# CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0 accelerate launch -m diffusion.pretrain.main \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_path "${CONFIG_PATH}" \
  --num_train_epochs ${EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size 1024 \
  --eval_steps 100 \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --dataloader_num_workers 0 \
  --max_steps 1000000 \
  --warmup_ratio 0.01 \
  --warmup_steps 2500 \
  --mlm_start_prob "${MLM_PROB_START}" \
  --mlm_end_prob "${MLM_PROB_END}" \
  --logging_steps 1 \
  --save_total_limit 2 \
  --seed 42 \
  --ema_decay 0 \
  --validation_dataset_name lm1b_test \
  --lr_scheduler_type constant_with_warmup \
  --max_length 128 \
  --mode "${MODE}" \
  --save_steps 1 \
  --bf16 \
  --pad_to_max_length
  # --lr_scheduler_kwargs "{'min_lr_rate': 0.005,'num_training_steps':281250,'num_warmup_steps':500}" \
  




