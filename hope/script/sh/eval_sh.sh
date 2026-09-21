#!/bin/bash
# eval_sh.sh - V2
# ==============================================================================
# Evaluation worker script - processes specific checkpoints passed as arguments
# ==============================================================================
# Usage:
#   bash eval_sh.sh <OUTPUT_DIR> <CHECKPOINT_LIST> <MODEL_TYPE> <BATCH_SIZE> \
#                <LIMIT> <DIFFUSION_EVAL_MODE> <TASKS_JSON> [DIFFUSION_TYPE] [MC_NUM]
#
# Arguments:
#   CHECKPOINT_LIST: Space-separated list of checkpoint paths, OR a parent directory
#                    (for backward compatibility, will auto-discover if directory)
# ==============================================================================

set -e

# --- Input Arguments ---
OUTPUT_DIR="$1"
CHECKPOINT_LIST_OR_DIR="$2"
BATCH_MODEL_TYPE="$3"
BATCH_SIZE="$4"
LIMIT="$5"
DIFFUSION_EVAL_MODE="$6"
TASKS_JSON="$7"
DIFFUSION_TYPE="${8:-mdlm}"
MC_NUM="${9:-32}"

# --- Validation ---
if [[ -z "$TASKS_JSON" ]]; then
    echo "Error: TASKS_JSON argument is missing."
    echo "Usage: bash eval.sh <OUTPUT_DIR> <CHECKPOINT_LIST> <MODEL_TYPE> <BATCH_SIZE> <LIMIT> <DIFFUSION_EVAL_MODE> <TASKS_JSON> [DIFFUSION_TYPE] [MC_NUM]"
    exit 1
fi

echo "=== Job Configuration ==="
echo "Output: $OUTPUT_DIR"
echo "Model Type: $BATCH_MODEL_TYPE"
echo "Batch Size: $BATCH_SIZE"
echo "Limit: $LIMIT"
echo "Diffusion Eval Mode: $DIFFUSION_EVAL_MODE"
echo "Diffusion Type: $DIFFUSION_TYPE"
echo "MC Num: $MC_NUM"
echo "Tasks: $TASKS_JSON"
echo "========================="

# --- Environment Setup ---
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/.bashrc
conda activate base
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion
echo "HF_HOME is set to: $HF_HOME"
export NUMEXPR_MAX_THREADS=16
export WANDB_DISABLED=true
# Detect correct network interface (route-based, avoids RDMA-only NIC like eth9)
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope/script/detect_nccl_ifname.sh

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,NET,GRAPH,ENV,DYNDBG
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=20
export NCCL_IB_RETRY_CNT=15
export NCCL_IB_QPS_PER_CONNECTION=8



# --- Parse Tasks from JSON ---
eval "$(python3 -c "
import json, sys, shlex
try:
    tasks = json.loads('${TASKS_JSON}')
    names = []
    shots = []
    for task, shot in tasks.items():
        names.append(shlex.quote(str(task)))
        shots.append(str(shot))
    print(f'TASK_NAMES=({\" \".join(names)})')
    print(f'TASK_SHOTS=({\" \".join(shots)})')
except Exception as e:
    print(f'echo Error parsing JSON: {e}', file=sys.stderr)
    sys.exit(1)
")"

echo "Parsed ${#TASK_NAMES[@]} task(s): ${TASK_NAMES[*]}"

# --- Model Discovery ---
# Handle both: direct checkpoint list OR parent directory (backward compatibility)
MODELS=()

# Check if input looks like a directory path (single path, exists as directory, no checkpoints inside the string)
if [[ -d "$CHECKPOINT_LIST_OR_DIR" && ! "$CHECKPOINT_LIST_OR_DIR" =~ [[:space:]] ]]; then
    # It's a single directory - check if it's a model or contains checkpoints
    if [ -f "${CHECKPOINT_LIST_OR_DIR}/config.json" ]; then
        # It's a model directory itself
        echo "Input is a model directory: $CHECKPOINT_LIST_OR_DIR"
        MODELS+=("$CHECKPOINT_LIST_OR_DIR")
    else
        # Try to discover checkpoints (backward compatibility mode)
        echo "Input is a parent directory. Discovering checkpoints..."
        while IFS= read -r cp_path; do
            if [ -n "$cp_path" ]; then
                MODELS+=("$cp_path")
                echo "  Found: $(basename "$cp_path")"
            fi
        done < <(find "$CHECKPOINT_LIST_OR_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V)
        
        # Fallback: use parent dir as model if no checkpoints found
        if [ ${#MODELS[@]} -eq 0 ] && [ -f "${CHECKPOINT_LIST_OR_DIR}/config.json" ]; then
            echo "No checkpoints found, using directory as model."
            MODELS+=("$CHECKPOINT_LIST_OR_DIR")
        fi
    fi
else
    # It's a space-separated list of checkpoint paths
    echo "Input is a checkpoint list. Processing..."
    for model_path in $CHECKPOINT_LIST_OR_DIR; do
        if [ -d "$model_path" ]; then
            MODELS+=("$model_path")
            echo "  Added: $(basename "$model_path")"
        else
            echo "  Warning: Path not found: $model_path"
        fi
    done
fi

# Verify we have models to process
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "ERROR: No valid models found!"
    echo "Input was: $CHECKPOINT_LIST_OR_DIR"
    exit 1
fi

echo "========================="
echo "Processing ${#MODELS[@]} model(s)"
echo "========================="

# --- Execution Loop ---
mkdir -p "$OUTPUT_DIR"

for i in "${!MODELS[@]}"; do
    model_path="${MODELS[$i]}"
    model_name=$(basename "$model_path")
    
    echo ""
    echo "=================================================================="
    echo "Model [$((i+1))/${#MODELS[@]}]: $model_name"
    echo "Path: $model_path"
    echo "=================================================================="
    
    # Handle folder naming logic
    if [[ "$model_name" == checkpoint-* ]]; then
        parent_name=$(basename "$(dirname "$model_path")")
        CURRENT_LOG_DIR="${OUTPUT_DIR}/${parent_name}/${model_name}"
    else
        CURRENT_LOG_DIR="${OUTPUT_DIR}/${model_name}"
    fi
    mkdir -p "$CURRENT_LOG_DIR"

    for j in "${!TASK_NAMES[@]}"; do
        t_name="${TASK_NAMES[$j]}"
        t_shot="${TASK_SHOTS[$j]}"
        log_file="${CURRENT_LOG_DIR}/log_${t_name}_${t_shot}shot.txt"

        echo "   Running Task: $t_name ($t_shot-shot)..."
        echo "   Log file: $log_file"
        
        # Build command with all arguments
        CMD="python3 -m eval.main \
            --model_name_or_path \"$model_path\" \
            --model_type \"$BATCH_MODEL_TYPE\" \
            --tasks \"$t_name\" \
            --num_fewshot \"$t_shot\" \
            --batch_size \"$BATCH_SIZE\" \
            --limit \"$LIMIT\" \
            --output_dir \"$OUTPUT_DIR\" \
            --diffusion_eval_mode \"$DIFFUSION_EVAL_MODE\" \
            --diffusion_type \"$DIFFUSION_TYPE\" \
            --mc_num \"$MC_NUM\" \
            --trust_remote_code --use_multi_gpu "
        
        echo "   Command: $CMD" >> "$log_file"
        
        set +e
        eval "$CMD" 2>&1 | tee -a "$log_file"
        exit_code=${PIPESTATUS[0]}
        set -e
        

        if [ $exit_code -ne 0 ]; then
            echo "    [FAILED] Exit code: $exit_code"
            echo "    See log: $log_file"
            # 关键修改：在这里显式退出脚本并返回错误码
            exit $exit_code
        else
            echo "    [SUCCESS]"
        fi

    done
done

echo ""
echo "=================================================================="
echo "Evaluation Complete!"
echo "=================================================================="
echo "Processed ${#MODELS[@]} model(s) with ${#TASK_NAMES[@]} task(s)"
echo "Results saved to: $OUTPUT_DIR"