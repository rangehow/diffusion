#!/bin/bash
# ============================================================================
# CARD-only efficiency benchmark
#
# Usage:
#     bash sh_benchmark_card_only.sh
#
# Runs ablation.benchmark_efficiency with only the CARD model (no AR baseline).
# ============================================================================

set -e

# Ensure we're in the project root
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion

echo "============================================================"
echo "CARD-only Efficiency Benchmark"
echo "============================================================"

# ==========================================
# Configuration
# ==========================================
CARD_MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_2node/checkpoint-309339"
OUTPUT_DIR="ablation_outputs/efficiency_card_only"
GPU_ID=0
TEST_ITER=10

mkdir -p "${OUTPUT_DIR}"

echo "Model:     ${CARD_MODEL_PATH}"
echo "Output:    ${OUTPUT_DIR}"
echo "GPU:       ${GPU_ID}"
echo "Test iter: ${TEST_ITER}"
echo "============================================================"

# ==========================================
# Run benchmark (CARD only, no --ar_model_path)
# ==========================================
python -m ablation.benchmark_efficiency \
    --card_model_path "${CARD_MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --gpu_id "${GPU_ID}" \
    --test_iter "${TEST_ITER}"

echo ""
echo "============================================================"
echo "CARD-only benchmark complete!"
echo "Results saved to: ${OUTPUT_DIR}/efficiency_benchmark.json"
echo "============================================================"
