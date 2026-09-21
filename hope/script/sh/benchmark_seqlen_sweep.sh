#!/bin/bash
# 序列长度 sweep 基准测试
# 参数与 benchmark_train.sh 相同，额外通过环境变量传入序列长度列表
# 每个 (model, seq_len) 组合跑 200 步

OUTPUT_DIR=$1
MLM_SCHEDULE_TYPE=$2
MLM_PROB_START=$3
MLM_PROB_END=$4
BATCH_SIZE=$5
GRAD_ACCUM_STEPS=$6
CONFIG_PATH=$7
MODE=${8}
DATASET_NAME=${9}
# MAX_LENGTH 由循环设置，不从参数读
EPOCHS=${11}
tail_bias_factor=${12}
use_daum=${13}
MAX_STEPS=${14:-200}

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

PAD_FLAG=""
if [ "${MODE}" = "bd3lm" ]; then
    PAD_FLAG="--pad_to_max_length"
fi

# 序列长度 sweep: 512, 1024, 2048, 4096
for SEQ_LEN in 512 1024 2048 4096; do
    SWEEP_OUTPUT="${OUTPUT_DIR}_seq${SEQ_LEN}"
    echo "=========================================="
    echo "  ${MODE} — seq_len=${SEQ_LEN}, ${MAX_STEPS} steps"
    echo "  Output: ${SWEEP_OUTPUT}"
    echo "=========================================="

    # 如果已完成则跳过
    if [ -f "${SWEEP_OUTPUT}/sweep_done" ]; then
        echo "已完成，跳过"
        continue
    fi

    # 清理旧的失败结果
    rm -rf "${SWEEP_OUTPUT}"

    ${launch} -m diffusion.pretrain.main \
      --model_name_or_path "${MODEL_PATH}" \
      --dataset_name "${DATASET_NAME}" \
      --output_dir "${SWEEP_OUTPUT}" \
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
      --save_steps 199 \
      --save_total_limit 1 \
      --seed 42 \
      --max_length "${SEQ_LEN}" \
      --mode "${MODE}" \
      --use_daum "${use_daum}" \
      --lr_scheduler_kwargs "{'min_lr_rate': 0.01}" \
      ${PAD_FLAG} \
      --bf16

    touch "${SWEEP_OUTPUT}/sweep_done"
    echo "${MODE} seq_len=${SEQ_LEN} 完成"
done

echo "全部序列长度 sweep 完成: ${MODE}"
