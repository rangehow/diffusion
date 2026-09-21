#!/bin/bash
# PrefixLM 1B training script
# Uses LLaDA backbone (bidirectional attention) with PrefixLM collator
# Aligned with MDLM main_exp: same model config, data, batch, steps

set -e

export NUMEXPR_MAX_THREADS=1000

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/.bashrc
conda activate base
echo "HF_HOME is set to: $HF_HOME"

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04
export WANDB_DISABLED=true

# --- Model & Data (aligned with MDLM main_exp) ---
MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/ModernBERT-base"
CONFIG_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config/llada_1b.json"
DATASET_NAME="sh_filtered_finefineweb"
OUTPUT_DIR="diffusion/model_output/prefixlm_main_exp"
MODE="prefixlm"
MAX_STEPS=${1:-500}
RESUME=${2:-false}          # pass 'true' as $2 to resume from checkpoint

# --- Hyperparameters (exactly matching MDLM main_exp) ---
MAX_LENGTH=2048
BATCH_SIZE=8
GRAD_ACCUM_STEPS=16
LEARNING_RATE=2e-4
EPOCHS=1
SEED=42

# --- NCCL ---
# Detect correct network interface (route-based, avoids RDMA-only NIC like eth9)
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope/script/detect_nccl_ifname.sh
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,NET,GRAPH,ENV,DYNDBG
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=8
# Socket-level retry — small safety net; the pre-launch barrier ensures
# all nodes are alive, so NCCL should connect on the first attempt.
export NCCL_SOCKET_RETRY_CNT=5
export NCCL_SOCKET_RETRY_SLEEP_MSEC=1000
# PyTorch-level heartbeat
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

# --- Clean up stale checkpoints unless explicitly resuming ---
if [ "${RESUME}" != "true" ]; then
  echo "RESUME=${RESUME} — cleaning old checkpoints in ${OUTPUT_DIR} ..."
  rm -rf "${OUTPUT_DIR}"/checkpoint-*
else
  echo "RESUME=true — will resume from latest checkpoint in ${OUTPUT_DIR}"
fi

# --- Pre-launch barrier: wait for ALL nodes to be alive BEFORE NCCL connects ---
python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope/script/wait_for_all_nodes.py
echo "All nodes confirmed alive, proceeding with launch."

launch=$(python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope/script/acc.py)
echo "Launch command: ${launch}"

${launch} -m diffusion.pretrain.main \
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
  --logging_steps 10 \
  --save_total_limit 2 \
  --seed ${SEED} \
  --max_length ${MAX_LENGTH} \
  --mode "${MODE}" \
  --lr_scheduler_kwargs "{'min_lr_rate': 0.01}" \
  --pad_to_max_length \
  --bf16 \
  --max_steps ${MAX_STEPS} \
  --save_strategy "steps" \
  --save_steps 500

echo "PrefixLM training complete."
