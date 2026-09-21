#!/bin/bash
# eval.sh
set -e  # Exit immediately if a command exits with a non-zero status

# --- Input Arguments ---
OUTPUT_DIR="$1"
TARGET_PARENT_DIR="$2"
BATCH_MODEL_TYPE="$3"
BATCH_SIZE="$4"
LIMIT="$5"
DIFFUSION_EVAL_MODE="$6"
TASKS_JSON="$7"

# --- Validation ---
if [[ -z "$TASKS_JSON" ]]; then
    echo "Error: TASKS_JSON argument is missing."
    exit 1
fi

echo "=== Job Configuration ==="
echo "Output: $OUTPUT_DIR"
echo "Target: $TARGET_PARENT_DIR"
echo "Tasks:  $TASKS_JSON"
echo "========================="

# --- Environment Setup ---
# Ensure these paths are accessible from the node running the job
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/.bashrc
mamba activate sglang || echo "Warning: Conda env activation failed, assuming env is pre-loaded."
export NUMEXPR_MAX_THREADS=16 # Reduced from 1000 to prevent CPU contention
export WANDB_DISABLED=true

# --- Parse Tasks from JSON ---
# We use Python to export arrays safe for Bash
eval "$(python3 -c "
import json, sys, shlex
try:
    tasks = json.loads('${TASKS_JSON}')
    names = []
    shots = []
    for task, shot in tasks.items():
        names.append(shlex.quote(str(task)))
        shots.append(str(shot))
    print(f'TASK_NAMES=({' '.join(names)})')
    print(f'TASK_SHOTS=({' '.join(shots)})')
except Exception as e:
    print(f'echo Error parsing JSON: {e}', file=sys.stderr)
    sys.exit(1)
")"

# --- Model Discovery ---
MODELS=()
MODEL_TYPES=()

if [ -d "$TARGET_PARENT_DIR" ]; then
    # Look for subdirectories starting with checkpoint-
    while IFS= read -r cp_path; do
        MODELS+=("$cp_path")
        MODEL_TYPES+=("$BATCH_MODEL_TYPE")
    done < <(find "$TARGET_PARENT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
fi

# Fallback: If no checkpoints found, check if the parent dir itself is the model
if [ ${#MODELS[@]} -eq 0 ]; then
    if [ -f "${TARGET_PARENT_DIR}/config.json" ]; then
        echo "No checkpoints found, using parent dir as model."
        MODELS+=("$TARGET_PARENT_DIR")
        MODEL_TYPES+=("$BATCH_MODEL_TYPE")
    else
        echo "ERROR: No models found in $TARGET_PARENT_DIR"
        exit 1
    fi
fi

# --- Execution Loop ---
mkdir -p "$OUTPUT_DIR"

for i in "${!MODELS[@]}"; do
    model_path="${MODELS[$i]}"
    model_type="${MODEL_TYPES[$i]}"
    model_name=$(basename "$model_path")
    
    # Handle folder naming logic
    if [[ "$model_name" == checkpoint-* ]]; then
        parent_name=$(basename "$(dirname "$model_path")")
        CURRENT_LOG_DIR="${OUTPUT_DIR}/${parent_name}/${model_name}"
    else
        CURRENT_LOG_DIR="${OUTPUT_DIR}/${model_name}"
    fi
    mkdir -p "$CURRENT_LOG_DIR"

    echo ">>> Processing Model: $model_name"

    for j in "${!TASK_NAMES[@]}"; do
        t_name="${TASK_NAMES[$j]}"
        t_shot="${TASK_SHOTS[$j]}"
        log_file="${CURRENT_LOG_DIR}/log_${t_name}_${t_shot}shot.txt"

        echo "   Running Task: $t_name ($t_shot-shot)"
        
        # We use 'python -m' and redirect output
        # NOTE: Added --trust_remote_code automatically
        python -m eval.main \
            --model_name_or_path "$model_path" \
            --model_type "$model_type" \
            --tasks "$t_name" \
            --num_fewshot "$t_shot" \
            --batch_size "$BATCH_SIZE" \
            --limit "$LIMIT" \
            --output_dir "$OUTPUT_DIR" \
            --diffusion_eval_mode "$DIFFUSION_EVAL_MODE" \
            --trust_remote_code
    done
done