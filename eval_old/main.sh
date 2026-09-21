#!/bin/bash
# main.sh

# ... (Keep your existing environment setup and exports) ...
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion
export NUMEXPR_MAX_THREADS=1000
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/.bashrc
proxy
mamba activate sglang
echo "Using HF_HOME: $HF_HOME"
export WANDB_DISABLED=true
set -e 

# ==========================================
# 1. BATCH CONFIGURATION
# ==========================================

# Initialize empty arrays
MODELS=()
MODEL_TYPES=()

# --- OPTION A: BATCH MODE (Auto-find checkpoints) ---
# Set the parent directory containing the checkpoint folders.
# Example: ".../model_output/arm_1B_100b_2node"
# TARGET_PARENT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/niu_1B_fineweb_edu_100b_potential"
# BATCH_MODEL_TYPE="causal" 

TARGET_PARENT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/mdlm_1B_fineweb_edu_100b_potential"
BATCH_MODEL_TYPE="causal"


# Logic to find and append checkpoints
if [ -d "$TARGET_PARENT_DIR" ]; then
    echo "--- Batch Mode Active ---"
    echo "Scanning: $TARGET_PARENT_DIR"
    
    # Find directories named "checkpoint-*", max depth 1, sort by version numbers (-V)
    # This ensures checkpoint-100 runs before checkpoint-1000
    while IFS= read -r cp_path; do
        MODELS+=("$cp_path")
        MODEL_TYPES+=("$BATCH_MODEL_TYPE")
        echo "Added: $(basename "$cp_path")"
    done < <(find "$TARGET_PARENT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
else
    echo "Target directory not found or empty, skipping batch discovery."
fi

# --- OPTION B: MANUAL MODE (Add specific models here if needed) ---
# You can append extra models manually if you want to test specific ones alongside the batch
# MODELS+=("/mnt/hdfs/.../Qwen2-0.5B/main")
# MODEL_TYPES+=("causal")


# Check if we have models to run
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "ERROR: No models found in $TARGET_PARENT_DIR and none defined manually."
    exit 1
fi

echo "Total models to evaluate: ${#MODELS[@]}"

# ==========================================
# 2. TASK CONFIGURATION
# ==========================================
declare -A TASK_SHOTS
TASK_SHOTS=(
    # ["mmlu"]=5
    ["hellaswag"]=10
    # ["arc_easy"]=25
    # ["arc_challenge"]=0
    # ["truthfulqa_mc2"]=0
    # ["piqa"]=0
    # ["sciq"]=0
    # ["commonsense_qa"]=5
)

# ==========================================
# 3. EXECUTION LOOP
# ==========================================
# ... (前面的配置保持不变) ...

# ==========================================
# 3. EXECUTION LOOP
# ==========================================

OUTPUT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/eval/evaluation_results"
BASE_BATCH_SIZE=16
LIMIT=0 
TRUST_REMOTE_CODE=true
TRUST_FLAG=""
if [ "$TRUST_REMOTE_CODE" = true ]; then
    TRUST_FLAG="--trust_remote_code"
fi

mkdir -p "$OUTPUT_DIR"

# Loop through all the models
for i in "${!MODELS[@]}"; do
    model_name="${MODELS[$i]}"
    model_type="${MODEL_TYPES[$i]}"
    
    echo "=================================================================="
    echo "Processing Model [$((i+1))/${#MODELS[@]}]: $model_name"
    echo "Type: $model_type"
    echo "=================================================================="

    batch_size=$BASE_BATCH_SIZE

    # --- 路径解析逻辑更新 ---
    # 获取文件名和父目录名
    model_basename=$(basename "$model_name")
    parent_name=$(basename "$(dirname "$model_name")")

    # 预先计算日志存放的具体目录，保持与 Python 代码一致的层级结构
    if [[ "$model_basename" == checkpoint-* ]]; then
        # 结构: results/Experiment/checkpoint-xxx/
        SPECIFIC_LOG_DIR="${OUTPUT_DIR}/${parent_name}/${model_basename}"
    else
        # 结构: results/ModelName/
        SPECIFIC_LOG_DIR="${OUTPUT_DIR}/${model_basename}"
    fi

    # 创建该目录以确保存放日志的位置存在
    mkdir -p "$SPECIFIC_LOG_DIR"

    # Inner loop for tasks
    for task_name in "${!TASK_SHOTS[@]}"; do
        num_shots="${TASK_SHOTS[$task_name]}"

        # 将日志文件直接保存在该模型/checkpoint的文件夹下，文件名前面加个 log_ 区分
        log_file="${SPECIFIC_LOG_DIR}/log_${task_name}_${num_shots}shot.txt"

        echo "  > Running Task: $task_name ($num_shots shot)..."

        set +e 
        
        # 注意：传给 Python 的 output_dir 依然是根目录 $OUTPUT_DIR
        # Python 脚本里的 prepare_output_env 会自动处理层级拼接
        python -m eval.main \
            --model_name_or_path "$model_name" \
            --model_type "$model_type" \
            --tasks "$task_name" \
            --num_fewshot "$num_shots" \
            --batch_size "$batch_size" \
            --limit "$LIMIT" \
            --output_dir "$OUTPUT_DIR" \
            $TRUST_FLAG > "$log_file" 2>&1

        exit_code=$?
        set -e 

        if [ $exit_code -ne 0 ]; then
            echo "    [FAILED] Check log: ${log_file}"
        else
            echo "    [SUCCESS] Log saved to: ${log_file}"
        fi
    done
done