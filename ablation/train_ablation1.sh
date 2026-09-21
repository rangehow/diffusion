#!/bin/bash
# ============================================================================
# Ablation 1: Component Decomposition — Train all 5 variants
#
# FAIRNESS GUARANTEE: All variants share:
#   - Same architecture:  ModernBertForDiffusionLM (NiuConfig, 1B params)
#   - Same init:          Random init with seed=42
#   - Same data:          fineweb_edu_1b (~1B tokens)
#   - Same hyperparams:   LR=5e-4, bs=4×64 accum, 50k steps, cosine LR
#   - Same tokenizer:     ModernBERT-base (50k vocab)
# Only the collator (masking/weighting strategy) changes.
# ============================================================================

set -e

# --- Paths (match pretrain/main.py conventions) ---
MODEL_PATH="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/answerdotai/ModernBERT-base/main"
CONFIG_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_1b.json"
DATASET_NAME="fineweb_edu_1b"
VALIDATION_DATASET_NAME="finefineweb_validation"
BASE_OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/ablation_outputs"

# --- Hyperparameters (must match your main CARD training exactly) ---
MAX_STEPS=50000
LR=5e-4
BATCH_SIZE=4
GRAD_ACCUM=64
MAX_LENGTH=2048
WARMUP_RATIO=0.01
SEED=42

# --- Working directory: project root ---
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion

VARIANTS=("full_card" "no_daum" "no_tail_bias" "no_both" "prefix_lm")

for VARIANT in "${VARIANTS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/ablation1_${VARIANT}"
    
    echo "============================================================"
    echo "Training variant: ${VARIANT}"
    echo "Output: ${OUTPUT_DIR}"
    echo "============================================================"
    
    accelerate launch ablation/train_ablation1.py \
        --variant "${VARIANT}" \
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
    
    echo "Variant ${VARIANT} training complete."
    echo ""
done

echo "============================================================"
echo "All ablation variants trained!"
echo "Results saved to: ${BASE_OUTPUT_DIR}"
echo "============================================================"
