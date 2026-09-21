#!/bin/bash
# sh_eval.sh - Worker script for distributed evaluation
# Usage: bash sh_eval.sh <CHECKPOINT_PATH> <MODE> <DATASET_NAME> <MAX_LENGTH> <BATCH_SIZE> <OUTPUT_DIR> [MAX_EVAL_SAMPLES] [USE_EMA]

set -e
CHECKPOINT_PATH=$1
MODE=${2:-"niu"}
DATASET_NAME=${3:-"sh_openwebtext"}
MAX_LENGTH=${4:-2048}
BATCH_SIZE=${5:-8}
OUTPUT_DIR=${6:-"./eval_results"}
MAX_EVAL_SAMPLES=${7:-""}
USE_EMA=${8:-"True"}  # 默认使用 EMA 权重

echo "--- 评估任务配置 ---"
echo "Checkpoint 路径: ${CHECKPOINT_PATH}"
echo "运行模式: ${MODE}"
echo "数据集名称: ${DATASET_NAME}"
echo "最大长度: ${MAX_LENGTH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "最大评估样本数: ${MAX_EVAL_SAMPLES:-'全部'}"
echo "使用 EMA 权重: ${USE_EMA}"
echo "----------------"

# --- 固定配置 ---
set -e

# 激活环境和设置环境变量
export NUMEXPR_MAX_THREADS=1000

source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/.bashrc
conda activate base
echo "HF_HOME is set to: $HF_HOME"

# 进入工作目录
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04
export WANDB_DISABLED=true

# 固定的路径
TOKENIZER_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/ModernBERT-base"

# 获取 accelerate launch 命令
launch=$(python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope/script/acc.py)
echo "Launch command: $launch"

# NCCL 配置
# export NCCL_SOCKET_IFNAME=br0
# export GLOO_SOCKET_IFNAME=br0 
# export TP_SOCKET_IFNAME=br0  

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,P2P,NET,GRAPH,ENV,DYNDBG
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_TIMEOUT=20
# export NCCL_IB_RETRY_CNT=15
# export NCCL_IB_QPS_PER_CONNECTION=8

# 构建评估命令参数
EVAL_ARGS=(
    --checkpoint_path "${CHECKPOINT_PATH}"
    --tokenizer_path "${TOKENIZER_PATH}"
    --dataset_name "${DATASET_NAME}"
    --mode "${MODE}"
    --max_length ${MAX_LENGTH}
    --per_device_eval_batch_size ${BATCH_SIZE}
    --output_dir "${OUTPUT_DIR}"
    --dataloader_num_workers 8
    --use_ema "${USE_EMA}"
    --pad_to_max_length
    --bf16
)

# 如果指定了 MAX_EVAL_SAMPLES，添加该参数
if [ -n "${MAX_EVAL_SAMPLES}" ]; then
    EVAL_ARGS+=(--max_eval_samples ${MAX_EVAL_SAMPLES})
fi

# 运行评估
${launch} -m diffusion.pretrain.eval_loss "${EVAL_ARGS[@]}"

echo "评估任务完成。"