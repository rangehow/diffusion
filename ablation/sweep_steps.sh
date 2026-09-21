#!/bin/bash
# ============================================================================
# Ablation 2 (Reviewer YsYm): Sensitivity to diffusion steps
# Fix block_size=16, sweep over number of steps
# ============================================================================

set -e

MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_2node/checkpoint-309339"
OUTPUT_DIR="ablation_outputs/sweep_steps"

python -m ablation.sweep_steps_and_blocks \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --block_sizes "16" \
    --step_counts "1,2,4,8,16,32,64" \
    --max_new_tokens 128 \
    --num_samples 1000 \
    --batch_size 64

echo "Step sweep complete. Now run PPL calculation:"
echo "python -m ppl.ppl_calculation --gen_dir ${OUTPUT_DIR} --ppl_model_path <reference_model>"
