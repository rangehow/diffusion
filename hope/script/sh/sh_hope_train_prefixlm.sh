#!/bin/bash
# =========================================================================
#  PrefixLM 1B — 测试运行 (1 node, 8 GPUs, 500 steps)
#  使用 LLaDA backbone (双向注意力) + PrefixLM collator
#  对齐 MDLM main_exp 的全部超参
#
#  沿用 sh_hope_train_4node.sh 的提交方式:
#    sed 替换 worker.script → hope run + -D 参数
# =========================================================================

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

HOPE_TEMPLATE="hope/benchmark_1node.hope"
TRAIN_SCRIPT="script/sh/train_prefixlm.sh"
MAX_STEPS=500

# --- 检查文件 ---
if [ ! -f "${HOPE_TEMPLATE}" ]; then
    echo "ERROR: HOPE template not found: ${HOPE_TEMPLATE}"
    exit 1
fi
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

echo "=== PrefixLM 1B Test Run ==="
echo "Nodes: 1 (8 GPUs)"
echo "bs=8, ga=16 => eff_batch = 8 * 16 * 8 = 1024 seqs/step"
echo "Steps: ${MAX_STEPS}"
echo "Config: llada_1b.json (same backbone as MDLM)"
echo "Data: sh_filtered_finefineweb"
echo "Output: diffusion/model_output/prefixlm_main_exp"
echo "==========================="

# 清理旧输出
rm -rf /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/prefixlm_main_exp

WORKER_CMD="bash ${TRAIN_SCRIPT} ${MAX_STEPS}"
TEMP_HOPE="prefixlm_1b_test.hope"
ESCAPED_CMD=$(printf '%s\n' "$WORKER_CMD" | sed -e 's/[\/&]/\\&/g')
sed "s#^worker.script = .*#worker.script = ${ESCAPED_CMD}#" "${HOPE_TEMPLATE}" > "${TEMP_HOPE}"

hope run "${TEMP_HOPE}" -Dhope.resource.experiment=fmg_h800_ci -Dmlp.sche.priority=P0
HOPE_EXIT=$?

rm -f "${TEMP_HOPE}"

if [ ${HOPE_EXIT} -eq 0 ]; then
    echo "PrefixLM test job submitted successfully."
else
    echo "ERROR: HOPE submission failed with exit code ${HOPE_EXIT}"
    exit ${HOPE_EXIT}
fi
