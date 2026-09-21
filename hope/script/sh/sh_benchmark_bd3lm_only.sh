#!/bin/bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

# Clean up old failed output
rm -rf /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/benchmark_bd3lm_1b

HOPE_TEMPLATE="hope/benchmark_1node.hope"

# BD3LM benchmark
hope run ${HOPE_TEMPLATE} \
    -e MAIN_SCRIPT_PATH=script/sh/benchmark_train.sh \
    -e MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/ModernBERT-base \
    -e CONFIG_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config/bd3lm_1b.json \
    -e DATASET=sh_filtered_finefineweb \
    -e MODE=bd3lm \
    -e OUTPUT_DIR=diffusion/model_output/benchmark_bd3lm_1b \
    -e BSZ=4 \
    -e GA=16 \
    -e MAX_LENGTH=2048 \
    -e TBF=1.5 \
    -e USE_DAUM=true \
    -e PAD="--pad_to_max_length" \
    -e MAX_STEPS=500

echo "BD3LM benchmark submitted"
