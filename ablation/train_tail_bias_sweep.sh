#!/bin/bash
# ============================================================================
# Ablation 5: Tail-bias factor sensitivity sweep
# Tests different tail_bias_factor values to show robustness.
#
# Uses the same train_ablation1.py script with --tail_bias_factor override.
# All variants use full_card (tail-biased + DAUM), only varying the factor.
# ============================================================================

set -e

MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/answerdotai/ModernBERT-base/main"
CONFIG_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_1b.json"
DATASET_NAME="fineweb_edu_1b"
VALIDATION_DATASET_NAME="finefineweb_validation"
BASE_OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/ablation_outputs"

MAX_STEPS=50000
LR=5e-4
BATCH_SIZE=4
GRAD_ACCUM=64
MAX_LENGTH=2048
WARMUP_RATIO=0.01
SEED=42

# Tail bias factors to test
# 1.0 = effectively no tail bias (CausalMLMCollator with factor=1.0 draws from full range)
# 1.5 = default CARD
# 2.0, 3.0 = stronger tail bias
TAIL_FACTORS=("1.0" "1.25" "1.5" "2.0" "3.0")

for TBF in "${TAIL_FACTORS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/ablation5_tbf_${TBF}"
    
    echo "============================================================"
    echo "Training with tail_bias_factor = ${TBF}"
    echo "Output: ${OUTPUT_DIR}"
    echo "============================================================"
    
    # Reuse train_ablation1.py with --tail_bias_factor override
    # cd to project root to match import paths
    cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion

    accelerate launch ablation/train_ablation1.py \
        --variant full_card \
        --tail_bias_factor "${TBF}" \
        --model_name_or_path "${MODEL_PATH}" \
        --config_path "${CONFIG_PATH}" \
        --dataset_name "${DATASET_NAME}" \
        --validation_dataset_name "${VALIDATION_DATASET_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --max_steps ${MAX_STEPS} \
        --learning_rate ${LR} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --max_length ${MAX_LENGTH} \
        --warmup_ratio ${WARMUP_RATIO} \
        --save_steps 10000 \
        --eval_steps 5000 \
        --seed ${SEED} \
        --pad_to_max_length \
        --lr_scheduler_kwargs "{'min_lr_rate': 0.01}"
    
    echo "tail_bias_factor=${TBF} training complete."
    echo ""
done

echo "============================================================"
echo "All tail_bias_factor sweep variants trained!"
echo "============================================================"
