#!/bin/bash
# =========================================================================
#  Reasoning Task Training Script
#  Follows Ye et al. (2024) "Beyond Autoregression" setting
#  Supports: CARD, AR, MDLM on Countdown (CD3/CD4/CD5) and Sudoku
#
#  Arguments:
#    $1  TASK_NAME    (cd3, cd4, cd5, sudoku)
#    $2  MODE         (reasoning_card, reasoning_ar, reasoning_mdlm)
#    $3  EPOCHS       (e.g., 600 for CD, 300 for Sudoku)
#    $4  BATCH_SIZE   (e.g., 128)
#    $5  MAX_LENGTH   (e.g., 64 for CD, 164 for Sudoku)
#    $6  LR           (e.g., 3e-4)
#    $7  MODEL_SIZE   (110m or 400m, default: 110m)
# =========================================================================

TASK_NAME=$1
MODE=$2
EPOCHS=${3:-600}
BATCH_SIZE=${4:-128}
MAX_LENGTH=${5:-128}
LR=${6:-3e-4}
MODEL_SIZE=${7:-110m}

echo "--- Reasoning Task Training ---"
echo "Task: ${TASK_NAME}"
echo "Mode: ${MODE}"
echo "Model Size: ${MODEL_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Length: ${MAX_LENGTH}"
echo "Learning Rate: ${LR}"
echo "-------------------------------"

set -e

export NUMEXPR_MAX_THREADS=1000

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/.bashrc
conda activate base
echo "HF_HOME is set to: $HF_HOME"

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04
export WANDB_DISABLED=true

# Fixed paths
BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04"
DIFFUSION_DIR="${BASE_DIR}/diffusion"

# Model path (tokenizer source)
MODEL_PATH="${BASE_DIR}/ModernBERT-base"

# Model config based on mode + size (reasoning-specific configs with vocab_size=22)
if [ "${MODE}" = "reasoning_ar" ]; then
    CONFIG_PATH="${DIFFUSION_DIR}/model_config/reasoning_llama_${MODEL_SIZE}.json"
elif [ "${MODE}" = "reasoning_mdlm" ] || [ "${MODE}" = "reasoning_mdm" ]; then
    CONFIG_PATH="${DIFFUSION_DIR}/model_config/reasoning_llada_${MODEL_SIZE}.json"
elif [ "${MODE}" = "reasoning_card" ]; then
    CONFIG_PATH="${DIFFUSION_DIR}/model_config/reasoning_modernbert_${MODEL_SIZE}.json"
else
    echo "ERROR: Unknown mode ${MODE}"
    exit 1
fi

# Verify config exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config not found: ${CONFIG_PATH}"
    exit 1
fi
echo "Model config: ${CONFIG_PATH}"

# Output directory
OUTPUT_DIR="${DIFFUSION_DIR}/model_output/reasoning_${TASK_NAME}_${MODE}_${MODEL_SIZE}"

# Dataset name (registered in load_dataset.py)
DATASET_NAME="${TASK_NAME}_train"

# Gradient accumulation to reach effective batch ~1024
# Detect number of GPUs from HOPE cluster spec (multi-node support)
NUM_WORKERS=$(python3 -c "
import os, json
spec = json.loads(os.environ.get('AFO_ENV_CLUSTER_SPEC', '{\"worker\":[\"localhost:0\"]}'))
role = spec.get('role', 'worker')
print(len(spec.get(role, ['localhost:0'])))
" 2>/dev/null || echo 1)
NUM_GPUS=$((NUM_WORKERS * 8))
echo "Detected ${NUM_WORKERS} node(s) x 8 GPUs = ${NUM_GPUS} total GPUs"
TARGET_EFF_BATCH=1024
GRAD_ACCUM_STEPS=$(( TARGET_EFF_BATCH / (NUM_GPUS * BATCH_SIZE) ))
if [ ${GRAD_ACCUM_STEPS} -lt 1 ]; then
    GRAD_ACCUM_STEPS=1
fi
echo "Gradient Accumulation Steps: ${GRAD_ACCUM_STEPS} (effective batch = $((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM_STEPS)))"

# --- Pre-launch barrier ---
python3 ${DIFFUSION_DIR}/hope/script/wait_for_all_nodes.py
echo "All nodes confirmed alive, proceeding with launch."

launch=$(python3 ${DIFFUSION_DIR}/hope/script/acc.py)
echo $launch

# NCCL interface detection
IF_NAME=$(python3 -c "
import os, json, socket, subprocess
spec = json.loads(os.environ['AFO_ENV_CLUSTER_SPEC'])
role = spec['role']
my_idx = int(spec['index'])
workers = spec[role]
peer_idx = (my_idx + 1) % len(workers)
peer_host = workers[peer_idx].split(':')[0]
peer_ip = socket.gethostbyname(peer_host)
out = subprocess.check_output(['ip', 'route', 'get', peer_ip], text=True)
tokens = out.split()
print(tokens[tokens.index('dev') + 1])
" 2>/dev/null)
IF_NAME=${IF_NAME:-eth0}
echo "Detected NCCL_SOCKET_IFNAME=${IF_NAME}"
export NCCL_SOCKET_IFNAME=${IF_NAME}
export GLOO_SOCKET_IFNAME=${IF_NAME}
export TP_SOCKET_IFNAME=${IF_NAME}

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,NET,GRAPH,ENV,DYNDBG
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_SOCKET_RETRY_CNT=5
export NCCL_SOCKET_RETRY_SLEEP_MSEC=1000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600

# Extra args for CARD
EXTRA_ARGS=""
if [ "${MODE}" = "reasoning_card" ]; then
    EXTRA_ARGS="--mlm_start_prob 1 --mlm_end_prob 0.0001 --tail_bias_factor 1.5 --use_daum True"
fi

${launch} -m diffusion.pretrain.main \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --config_path "${CONFIG_PATH}" \
    --num_train_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --dataloader_num_workers 8 \
    --warmup_ratio 0.01 \
    --logging_steps 10 \
    --save_total_limit 2 \
    --save_steps 5000 \
    --seed 42 \
    --max_length "${MAX_LENGTH}" \
    --mode "${MODE}" \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs "{'min_lr_rate': 0.01}" \
    --pad_to_max_length \
    --bf16 \
    ${EXTRA_ARGS}

echo "Training completed for ${TASK_NAME} (${MODE}) at ${MODEL_SIZE}."
