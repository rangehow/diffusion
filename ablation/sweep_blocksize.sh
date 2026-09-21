#!/bin/bash
# ============================================================================
# Ablation 3 (Reviewer mkMw): Block size vs quality trade-off
# Fix steps=8, sweep over block_size
# ============================================================================

set -e

MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_2node/checkpoint-309339"
OUTPUT_DIR="ablation_outputs/sweep_blocksize"

python -m ablation.sweep_steps_and_blocks \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --block_sizes "1,2,4,8,16,32,64,128" \
    --step_counts "8" \
    --max_new_tokens 128 \
    --num_samples 1000 \
    --batch_size 64

echo "Block size sweep complete. Now run PPL calculation:"
echo "python -m ppl.ppl_calculation --gen_dir ${OUTPUT_DIR} --ppl_model_path <reference_model>"
