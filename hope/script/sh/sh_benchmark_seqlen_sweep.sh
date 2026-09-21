#!/bin/bash
# =========================================================================
#  序列长度 Sweep 基准测试 — 4个模型各跑 512/1024/2048/4096
#  每个模型 1 node (8 GPUs), 每个 seq_len 200 步
#  4个模型并行提交，每个模型内部串行跑4个 seq_len
#
#  使用 sed 替换 worker.script 的方式提交 (与其他 HOPE 脚本一致)
# =========================================================================

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

HOPE_TEMPLATE="hope/benchmark_1node.hope"
TRAIN_SCRIPT="script/sh/benchmark_seqlen_sweep.sh"
BASE_OUTPUT="diffusion/model_output"
MAX_STEPS=200

# 超参全部对齐: bs=4, ga=16, 8 GPUs → eff_batch=512
BS=4
GA=16

CONFIG_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config"

# --- 检查文件 ---
if [ ! -f "${HOPE_TEMPLATE}" ]; then
    echo "ERROR: Template not found: ${HOPE_TEMPLATE}"
    exit 1
fi
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

echo "=========================================="
echo "  序列长度 Sweep 基准测试"
echo "  4 models × 4 seq_lens × ${MAX_STEPS} steps"
echo "  bs=${BS}, ga=${GA}, 8 GPUs"
echo "=========================================="

# =========================================================================
#  submit_sweep: 提交一个模型的 seq_len sweep
# =========================================================================
submit_sweep() {
    local JOB_NAME=$1
    local OUTPUT_SUBDIR=$2
    local MODE=$3
    local CONFIG=$4
    local TBF=$5
    local DAUM=$6

    local FULL_OUTPUT="${BASE_OUTPUT}/${OUTPUT_SUBDIR}"

    # benchmark_seqlen_sweep.sh 内部循环 seq_len，MAX_LENGTH 占位传 2048
    local WORKER_CMD="bash ${TRAIN_SCRIPT} \
        \"${FULL_OUTPUT}\" \
        \"cosine_with_min_lr\" \
        \"1\" \
        \"0.0001\" \
        \"${BS}\" \
        \"${GA}\" \
        \"${CONFIG}\" \
        \"${MODE}\" \
        \"sh_filtered_finefineweb\" \
        \"2048\" \
        \"1\" \
        \"${TBF}\" \
        \"${DAUM}\" \
        \"${MAX_STEPS}\""

    echo "=== Submitting: ${JOB_NAME} ==="
    echo "  Mode: ${MODE}, BS=${BS}, GA=${GA}"
    echo "  Output: ${FULL_OUTPUT}_seq{512,1024,2048,4096}"

    local TEMP_HOPE="${JOB_NAME}.hope"
    local ESCAPED_CMD
    ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
    sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

    hope run "${TEMP_HOPE}" -Dhope.resource.experiment=fmg_h800_ci -Dmlp.sche.priority=P0 &
    local EXIT_CODE=$?
    echo "  Submitted (PID: $!)"

    TEMP_FILES+=("${TEMP_HOPE}")
}

declare -a TEMP_FILES=()

# --- 提交 4 个模型 (并行) ---

# 1. CARD (ModernBERT, causal, unpad)
submit_sweep "sweep_card" "sweep_card_1b" "niu" \
    "${CONFIG_DIR}/modernbert_1b.json" "100" "True"

# 2. ARM (LLaMA, causal)
submit_sweep "sweep_arm" "sweep_arm_1b" "llama" \
    "${CONFIG_DIR}/llama_1b.json" "100" "False"

# 3. MDLM (LLaDA, bidirectional)
submit_sweep "sweep_mdlm" "sweep_mdlm_1b" "mdlm" \
    "${CONFIG_DIR}/llada_1b.json" "100" "False"

# 4. BD3LM (block diffusion, padded)
submit_sweep "sweep_bd3lm" "sweep_bd3lm_1b" "bd3lm" \
    "${CONFIG_DIR}/bd3lm_1b.json" "100" "False"

# --- 等待 + 清理 ---
echo ""
echo "等待所有 HOPE 提交完成..."
wait

echo "清理临时文件..."
for f in "${TEMP_FILES[@]}"; do
    rm -f "$f"
done

echo ""
echo "=========================================="
echo "  全部 sweep 任务已提交！"
echo "  每个模型跑 4 个 seq_len × ${MAX_STEPS} 步"
echo "  预计 1-2 小时完成"
echo "  完成后: python3 hope/script/sh/analyze_seqlen_sweep.py"
echo "=========================================="
