#!/bin/bash
# =========================================================================
#  Evaluate all trained reasoning task models
#
#  Usage:
#    cd /path/to/diffusion
#    bash hope/script/sh/eval_reasoning.sh [--size 110m|400m|both]
# =========================================================================

set -e

SIZE_FILTER="both"
while [[ $# -gt 0 ]]; do
    case $1 in
        --size) SIZE_FILTER="$2"; shift 2 ;;
        *)      echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04"
DIFFUSION_DIR="${BASE_DIR}/diffusion"
OUTPUT_DIR="${DIFFUSION_DIR}/reasoning_tasks/eval_output"
TOKENIZER_PATH="${BASE_DIR}/ModernBERT-base"

cd ${BASE_DIR}

mkdir -p ${OUTPUT_DIR}

# Determine sizes
SIZES=()
if [ "${SIZE_FILTER}" = "110m" ]; then
    SIZES=("110m")
elif [ "${SIZE_FILTER}" = "400m" ]; then
    SIZES=("400m")
elif [ "${SIZE_FILTER}" = "both" ]; then
    SIZES=("110m" "400m")
fi

evaluate_model() {
    local TASK=$1
    local MODE=$2
    local MAX_NEW_TOKENS=$3
    local NUM_STEPS=$4
    local BLOCK_SIZE=$5
    local MODEL_SIZE=$6
    
    MODEL_DIR="${DIFFUSION_DIR}/model_output/reasoning_${TASK}_${MODE}_${MODEL_SIZE}"
    
    # Find latest checkpoint
    CKPT=$(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    
    if [ -z "${CKPT}" ]; then
        if [ -f "${MODEL_DIR}/config.json" ]; then
            CKPT="${MODEL_DIR}"
        else
            echo "WARNING: No checkpoint found for ${TASK} x ${MODE} x ${MODEL_SIZE}, skipping"
            return
        fi
    fi
    
    echo "================================================"
    echo "Evaluating: ${TASK} x ${MODE} x ${MODEL_SIZE}"
    echo "Checkpoint: ${CKPT}"
    echo "================================================"
    
    python -m diffusion.reasoning_tasks.evaluate \
        --model_path "${CKPT}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --task ${TASK} \
        --mode ${MODE} \
        --data_dir "${DIFFUSION_DIR}/reasoning_tasks/data" \
        --output_file "${OUTPUT_DIR}/${TASK}_${MODE}_${MODEL_SIZE}.json" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --num_steps ${NUM_STEPS} \
        --block_size ${BLOCK_SIZE} \
        --temperature 0.0 \
        --device cuda
    
    echo ""
}

for MODEL_SIZE in "${SIZES[@]}"; do
    echo "###################################################################"
    echo "# Evaluating Model Size: ${MODEL_SIZE}"
    echo "###################################################################"
    
    # Countdown tasks
    for TASK in cd3 cd4 cd5; do
        evaluate_model ${TASK} reasoning_card  50  16  32  ${MODEL_SIZE}
        evaluate_model ${TASK} reasoning_ar    50  0   0   ${MODEL_SIZE}
        evaluate_model ${TASK} reasoning_mdlm  50  64  0   ${MODEL_SIZE}
    done

    # Sudoku
    evaluate_model sudoku reasoning_card  100  32  96  ${MODEL_SIZE}
    evaluate_model sudoku reasoning_ar    100  0   0   ${MODEL_SIZE}
    evaluate_model sudoku reasoning_mdlm  100  64  0   ${MODEL_SIZE}
done

echo ""
echo "==========================================="
echo "All evaluations complete!"
echo "==========================================="

# Aggregate results
echo ""
echo "Generating results table..."
python -m diffusion.reasoning_tasks.aggregate_results --results_dir ${OUTPUT_DIR}
