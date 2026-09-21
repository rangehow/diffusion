#!/bin/bash
# 这个脚本现在接收10个参数来运行训练任务



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

echo "--- 任务配置 ---"
echo "输出目录: ${OUTPUT_DIR}"
echo "MLM 调度器: ${MLM_SCHEDULE_TYPE}"
echo "MLM 概率起始值: ${MLM_PROB_START}"
echo "MLM 概率结束值: ${MLM_PROB_END}"
echo "Batch Size: ${BATCH_SIZE}"
echo "梯度累积步数: ${GRAD_ACCUM_STEPS}"
echo "模型配置文件: ${CONFIG_PATH}"
echo "运行模式: ${MODE}"
echo "数据集名称: ${DATASET_NAME}"
echo "最大长度: ${MAX_LENGTH}"
echo "----------------"





# --- 固定配置 ---
# 确保脚本在遇到错误时退出
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
MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/ModernBERT-base"
# 固定的训练超参数

LEARNING_RATE=2e-4

# --- Pre-launch barrier: wait for ALL nodes to be alive BEFORE NCCL connects ---
python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope/script/wait_for_all_nodes.py
echo "All nodes confirmed alive, proceeding with launch."

launch=$(python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope/script/acc.py)
# launch=$(python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/hope/torch_run.py)
echo $launch

# NCCL — detect the correct network interface by finding which one routes to
# our peer nodes.  The old "first eth*" heuristic picked eth9 (the RDMA/RoCE
# NIC) which does NOT allow TCP connections — only RDMA verbs. NCCL's TCP
# bootstrap needs a regular TCP-capable interface.
IF_NAME=$(python3 -c "
import os, json, socket, subprocess
spec = json.loads(os.environ['AFO_ENV_CLUSTER_SPEC'])
role = spec['role']
my_idx = int(spec['index'])
workers = spec[role]
# Pick a peer node (not ourselves) to test routing
peer_idx = (my_idx + 1) % len(workers)
peer_host = workers[peer_idx].split(':')[0]
peer_ip = socket.gethostbyname(peer_host)
# Ask kernel which interface routes to that IP
out = subprocess.check_output(['ip', 'route', 'get', peer_ip], text=True)
tokens = out.split()
print(tokens[tokens.index('dev') + 1])
" 2>/dev/null)
IF_NAME=${IF_NAME:-eth0}
echo "Detected NCCL_SOCKET_IFNAME=${IF_NAME}  (route-based, avoids RDMA-only NIC)"
export NCCL_SOCKET_IFNAME=${IF_NAME}
export GLOO_SOCKET_IFNAME=${IF_NAME}
export TP_SOCKET_IFNAME=${IF_NAME}

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


${launch} -m  diffusion.pretrain.main \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_path "${CONFIG_PATH}" \
  --num_train_epochs ${EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --tail_bias_factor ${tail_bias_factor} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --dataloader_num_workers 8 \
  --warmup_ratio 0.01 \
  --mlm_start_prob "${MLM_PROB_START}" \
  --mlm_end_prob "${MLM_PROB_END}" \
  --logging_steps 10 \
  --save_total_limit 2 \
  --seed 42 \
  --max_length "${MAX_LENGTH}" \
  --mode "${MODE}" \
  --use_daum "${use_daum}" \
  --lr_scheduler_kwargs "{'min_lr_rate': 0.01}" \
  --pad_to_max_length \
  --bf16

echo "训练任务完成。"