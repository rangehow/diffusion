#!/bin/bash
# ============================================================
# Qwen3-8B Continued Pretraining — worker script
# Called by HOPE with positional arguments.
# ============================================================

MODE=$1
OUTPUT_DIR=$2
DATASET_NAME=$3
MAX_LENGTH=$4
BATCH_SIZE=$5
GRAD_ACCUM_STEPS=$6
LEARNING_RATE=$7
EXTRA_ARGS=$8  # Additional args as a single string

echo "============================================================"
echo " Qwen3-8B CPT — Mode: ${MODE}"
echo " Output: ${OUTPUT_DIR}"
echo " Dataset: ${DATASET_NAME}"
echo " Max Length: ${MAX_LENGTH}"
echo " Batch Size: ${BATCH_SIZE}"
echo " Grad Accum: ${GRAD_ACCUM_STEPS}"
echo " LR: ${LEARNING_RATE}"
echo " Extra: ${EXTRA_ARGS}"
echo "============================================================"

set -e

# ── Environment ──
export NUMEXPR_MAX_THREADS=1000
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/.bashrc
mamba activate sglang
echo "HF_HOME=$HF_HOME"

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04
export WANDB_DISABLED=true

# ── Fixed paths ──
MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/shxs/model/Qwen/Qwen3-8B-Base/main"

# ── Accelerate launch ──
launch=$(python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/hope/script/acc.py)
echo "Launch command: ${launch}"

${launch} -m diffusion.pretrain.qwen3_cpt \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --mode "${MODE}" \
  --max_length ${MAX_LENGTH} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --learning_rate ${LEARNING_RATE} \
  --num_train_epochs 1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --eval_steps 500 \
  --dataloader_num_workers 8 \
  --seed 42 \
  --bf16 \
  --gradient_checkpointing \
  --pad_to_max_length \
  ${EXTRA_ARGS}

echo "Training complete: ${MODE}"
