#!/bin/bash
# eval/scripts/run_eval.sh
#
# Example batch evaluation script using the refactored framework.
# This script mirrors the functionality of your original main.sh

set -e

# ==========================================
# Environment Setup
# ==========================================

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/.bashrc
mamba activate sglang
# proxy


export NUMEXPR_MAX_THREADS=1000
export WANDB_DISABLED=true

echo "=================================="
echo "Evaluation Framework v2.0"
echo "=================================="

# ==========================================
# Configuration
# ==========================================

# Model discovery settings
TARGET_PARENT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_2node/checkpoint-309339"
BATCH_MODEL_TYPE="causal"

# Initialize model arrays
MODELS=()
MODEL_TYPES=()

# Auto-discover checkpoints
if [ -d "$TARGET_PARENT_DIR" ]; then
    echo "Scanning for checkpoints: $TARGET_PARENT_DIR"
    
    while IFS= read -r cp_path; do
        MODELS+=("$cp_path")
        MODEL_TYPES+=("$BATCH_MODEL_TYPE")
        echo "  Found: $(basename "$cp_path")"
    done < <(find "$TARGET_PARENT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
fi

# Manual model additions (optional)
# MODELS+=("/path/to/specific/model")
# MODEL_TYPES+=("causal")

# Check if we have models
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "ERROR: No models found!"
    exit 1
fi

echo "Total models to evaluate: ${#MODELS[@]}"

# ==========================================
# Task Configuration
# ==========================================
# Format: task_name=num_fewshot
declare -A TASK_SHOTS
TASK_SHOTS=(
    ["hellaswag"]=3
    # ["mmlu"]=5
    # ["arc_easy"]=25
    # ["arc_challenge"]=25
    # ["piqa"]=0
    # ["winogrande"]=5
    # ["commonsense_qa"]=7
    # ["truthfulqa_mc2"]=0
    # ["sciq"]=0
)

# ==========================================
# Evaluation Settings
# ==========================================
OUTPUT_DIR="./evaluation_results"
BATCH_SIZE=32
LIMIT=0  # 0 = no limit
TRUST_REMOTE_CODE=true
NUM_WORKERS=8

# Diffusion-specific settings
DIFFUSION_EVAL_MODE="mc"   # "mc" (Monte Carlo) or "pll" (Pseudo-Log-Likelihood)
DIFFUSION_TYPE="causal"      # "causal", "mdlm", or "bd3lm"
MC_NUM=128                 # Number of Monte Carlo samples (only used when mode=mc)
MC_BATCH_SIZE=16           # Batch size for MC sampling
BLOCK_SIZE=16              # Block size for BD3LM (only used when diffusion_type=bd3lm)

# Build trust flag
TRUST_FLAG=""
if [ "$TRUST_REMOTE_CODE" = true ]; then
    TRUST_FLAG="--trust_remote_code"
fi

mkdir -p "$OUTPUT_DIR"

# ==========================================
# Execution Loop
# ==========================================
for i in "${!MODELS[@]}"; do
    model_name="${MODELS[$i]}"
    model_type="${MODEL_TYPES[$i]}"
    
    echo ""
    echo "=================================================================="
    echo "Model [$((i+1))/${#MODELS[@]}]: $(basename "$model_name")"
    echo "Type: $model_type"
    echo "=================================================================="
    
    # Prepare log directory
    model_basename=$(basename "$model_name")
    parent_name=$(basename "$(dirname "$model_name")")
    
    if [[ "$model_basename" == checkpoint-* ]]; then
        LOG_DIR="${OUTPUT_DIR}/${parent_name}/${model_basename}"
    else
        LOG_DIR="${OUTPUT_DIR}/${model_basename}"
    fi
    mkdir -p "$LOG_DIR"
    
    # Build diffusion-specific arguments
    DIFFUSION_ARGS=""
    if [ "$model_type" = "discrete_diffusion" ]; then
        DIFFUSION_ARGS="--diffusion_eval_mode $DIFFUSION_EVAL_MODE"
        DIFFUSION_ARGS="$DIFFUSION_ARGS --diffusion_type $DIFFUSION_TYPE"
        if [ "$DIFFUSION_EVAL_MODE" = "mc" ]; then
            DIFFUSION_ARGS="$DIFFUSION_ARGS --mc_num $MC_NUM"
            DIFFUSION_ARGS="$DIFFUSION_ARGS --mc_batch_size $MC_BATCH_SIZE"
        fi
        if [ "$DIFFUSION_TYPE" = "bd3lm" ]; then
            DIFFUSION_ARGS="$DIFFUSION_ARGS --block_size $BLOCK_SIZE"
        fi
    fi
    
    # Run each task
    for task_name in "${!TASK_SHOTS[@]}"; do
        num_shots="${TASK_SHOTS[$task_name]}"
        log_file="${LOG_DIR}/log_${task_name}_${num_shots}shot.txt"
        
        echo "  > Task: $task_name ($num_shots-shot)..."
        
        set +e
        python -m eval.main \
            --model_name_or_path "$model_name" \
            --model_type "$model_type" \
            --tasks "$task_name" \
            --num_fewshot "$num_shots" \
            --batch_size "$BATCH_SIZE" \
            --limit "$LIMIT" \
            --num_workers "$NUM_WORKERS" \
            --output_dir "$OUTPUT_DIR" \
            $DIFFUSION_ARGS \
            $TRUST_FLAG 
        
        exit_code=$?
        set -e
        
        if [ $exit_code -ne 0 ]; then
            echo "    [FAILED] See: $log_file"
        else
            echo "    [SUCCESS] Log: $log_file"
        fi
    done
done

echo ""
echo "=================================="
echo "Evaluation Complete!"
echo "=================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To generate Excel summary:"
echo "  python -m eval.export_results --results_dir $OUTPUT_DIR"