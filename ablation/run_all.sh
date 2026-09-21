#!/bin/bash
# ============================================================================
# Master script: Run ALL ablation experiments
#
# This script runs ablations in order of priority based on reviewer requests.
# Each section can be run independently.
# ============================================================================

set -e

# Ensure we're in the project root
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion

echo "============================================================"
echo "CARD Ablation Experiments - Full Pipeline"
echo "============================================================"
echo ""

# Common paths (modify these for your environment)
CARD_MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_2node/checkpoint-309339"
AR_MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/arm_main_exp"
GEN_OUTPUT_DIR="ppl/generation_outputs"

BASE_OUTPUT="ablation_outputs"
mkdir -p ${BASE_OUTPUT}

# ============================================================================
# PHASE 1: Training-based ablations (long-running, start first)
# ============================================================================

echo ""
echo "================================================================"
echo "PHASE 1: Training ablation variants (Reviewer 2Bg4)"
echo "================================================================"
# Uncomment when ready to run (these take many hours)
# bash ablation/train_ablation1.sh

echo ""
echo "================================================================"
echo "PHASE 1b: Tail-bias factor sensitivity (Ablation 5)"
echo "================================================================"
# bash ablation/train_tail_bias_sweep.sh


# ============================================================================
# PHASE 2: Inference-based ablations (can run on existing checkpoints)
# ============================================================================

echo ""
echo "================================================================"
echo "PHASE 2a: Diffusion steps sensitivity sweep (Reviewer YsYm)"
echo "================================================================"
python -m ablation.sweep_steps_and_blocks \
    --model_path "${CARD_MODEL_PATH}" \
    --output_dir "${BASE_OUTPUT}/sweep_steps" \
    --block_sizes "16" \
    --step_counts "1,2,4,8,16,32,64" \
    --max_new_tokens 128 \
    --num_samples 1000 \
    --batch_size 64

echo ""
echo "================================================================"
echo "PHASE 2b: Block size vs quality sweep (Reviewer mkMw)"
echo "================================================================"
python -m ablation.sweep_steps_and_blocks \
    --model_path "${CARD_MODEL_PATH}" \
    --output_dir "${BASE_OUTPUT}/sweep_blocksize" \
    --block_sizes "1,2,4,8,16,32,64,128" \
    --step_counts "8" \
    --max_new_tokens 128 \
    --num_samples 1000 \
    --batch_size 64


# ============================================================================
# PHASE 3: Efficiency benchmark (Reviewer Ruxi)
# ============================================================================

echo ""
echo "================================================================"
echo "PHASE 3: Efficiency benchmark (CARD vs AR)"
echo "================================================================"
python -m ablation.benchmark_efficiency \
    --card_model_path "${CARD_MODEL_PATH}" \
    --ar_model_path "${AR_MODEL_PATH}" \
    --output_dir "${BASE_OUTPUT}/efficiency" \
    --test_iter 10


# ============================================================================
# PHASE 4: Per-model PPL (Reviewer 2Bg4)
# ============================================================================

echo ""
echo "================================================================"
echo "PHASE 4: Per-model Gen PPL (separate reference models)"
echo "================================================================"
python -m ablation.ppl_per_model \
    --gen_dir "${GEN_OUTPUT_DIR}" \
    --output_dir "${BASE_OUTPUT}/ppl_per_model" \
    --batch_size 4


# ============================================================================
# PHASE 5: Aggregate all results
# ============================================================================

echo ""
echo "================================================================"
echo "PHASE 5: Aggregate results and generate figures"
echo "================================================================"
python -m ablation.aggregate_results \
    --ablation1_dir "${BASE_OUTPUT}" \
    --sweep_dir "${BASE_OUTPUT}/sweep_steps" \
    --efficiency_dir "${BASE_OUTPUT}/efficiency" \
    --output_dir "${BASE_OUTPUT}/figures"

# Also aggregate blocksize sweep
python -m ablation.aggregate_results \
    --sweep_dir "${BASE_OUTPUT}/sweep_blocksize" \
    --output_dir "${BASE_OUTPUT}/figures_blocksize"


echo ""
echo "============================================================"
echo "ALL ABLATION EXPERIMENTS COMPLETE!"
echo "============================================================"
echo "Results are in: ${BASE_OUTPUT}/"
echo "Figures are in: ${BASE_OUTPUT}/figures/"
echo "============================================================"
