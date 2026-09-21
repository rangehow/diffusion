#!/bin/bash
# =========================================================================
#  速度基准测试 — 1 node (8 GPUs), 500 steps
#  严格对齐: bs=4, ga=16, 8 GPUs → eff_batch = 512 seqs/step (全部相同)
#
#  沿用 sh_hope_train_4node.sh 的提交方式:
#    sed 替换 worker.script → hope run + -D 参数
# =========================================================================

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

HOPE_TEMPLATE="hope/benchmark_1node.hope"
MAIN_SCRIPT_PATH="script/sh/benchmark_train.sh"
BASE_OUTPUT_DIR="diffusion/model_output"
MAX_STEPS=500

CONFIG_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config"

# --- 检查模板和脚本 ---
if [ ! -f "${HOPE_TEMPLATE}" ]; then
    echo "ERROR: Template not found: ${HOPE_TEMPLATE}"
    exit 1
fi
if [ ! -f "${MAIN_SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${MAIN_SCRIPT_PATH}"
    exit 1
fi

# =========================================================================
#  submit_job: 复用 sh_hope_train_4node.sh 的 sed + hope run 逻辑
# =========================================================================
submit_job() {
    local JOB_NAME=$1
    local OUTPUT_SUBDIR=$2
    local MODE=$3
    local CONFIG=$4
    local MLM_START=$5
    local MLM_END=$6
    local TBF=$7
    local DAUM=$8

    local FULL_OUTPUT="${BASE_OUTPUT_DIR}/${OUTPUT_SUBDIR}"

    local WORKER_CMD="bash ${MAIN_SCRIPT_PATH} \
        \"${FULL_OUTPUT}\" \
        \"cosine_with_min_lr\" \
        \"${MLM_START}\" \
        \"${MLM_END}\" \
        \"4\" \
        \"16\" \
        \"${CONFIG}\" \
        \"${MODE}\" \
        \"sh_filtered_finefineweb\" \
        \"2048\" \
        \"1\" \
        \"${TBF}\" \
        \"${DAUM}\" \
        \"${MAX_STEPS}\""

    echo "=== Submitting: ${JOB_NAME} ==="
    echo "  Mode: ${MODE}, BS=4, GA=16, EffBatch=512"
    echo "  Output: ${FULL_OUTPUT}"

    local TEMP_HOPE="${JOB_NAME}.hope"
    local ESCAPED_CMD
    ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
    sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

    hope run "${TEMP_HOPE}" -Dhope.resource.experiment=fmg_h800_ci -Dmlp.sche.priority=P0
    local EXIT_CODE=$?
    echo "  Submitted (exit: ${EXIT_CODE})"

    rm -f "${TEMP_HOPE}"

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  WARNING: hope run exited with ${EXIT_CODE}"
    fi
}

# =========================================================================
#  提交 4 个任务 (bs=4, ga=16 完全对齐)
# =========================================================================

# 清理旧输出
rm -rf /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/${BASE_OUTPUT_DIR}/benchmark_card_1b
rm -rf /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/${BASE_OUTPUT_DIR}/benchmark_arm_1b
rm -rf /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/${BASE_OUTPUT_DIR}/benchmark_mdlm_1b
rm -rf /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/${BASE_OUTPUT_DIR}/benchmark_bd3lm_1b

# CARD
submit_job "benchmark_card" "benchmark_card_1b" "niu" \
    "${CONFIG_DIR}/modernbert_1b.json" "1" "0.0001" "1.5" "True"

# ARM
submit_job "benchmark_arm" "benchmark_arm_1b" "llama" \
    "${CONFIG_DIR}/llama_1b.json" "0.9" "0.1" "1.5" "True"

# MDLM
submit_job "benchmark_mdlm" "benchmark_mdlm_1b" "mdlm" \
    "${CONFIG_DIR}/llada_1b.json" "1" "0.0001" "1.5" "True"

# BD3LM
submit_job "benchmark_bd3lm" "benchmark_bd3lm_1b" "bd3lm" \
    "${CONFIG_DIR}/bd3lm_1b.json" "1" "0.0001" "1.5" "True"

echo ""
echo "=== All 4 benchmark jobs submitted ==="
echo "完成后运行: python3 hope/script/sh/analyze_benchmark.py"
