#!/bin/bash
# hope_eval.sh
# ==============================================================================
# HOPE 评估任务提交脚本
# ==============================================================================

# ==============================================================================
# 1. 定义默认参数
# ==============================================================================
declare -A default_config=(
    [batch_model_type]="discrete_diffusion"
    [batch_size]=16
    [limit]=0
    [diffusion_eval_mode]="mc"
    [tasks_json]='{"hellaswag":3}'
)

# ==============================================================================
# 2. 定义任务特定的覆盖参数
# ==============================================================================
task_overrides=(
    '([hope_name]="eval_mdlm_1B_fineweb" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/mdlm_1B_fineweb_edu_100b_potential" [batch_model_type]="discrete_diffusion" [tasks_json]="{\"hellaswag\":3}")'
    # '([hope_name]="eval_llada_1B" [target_parent_dir]="/path/to/llada_model" [batch_model_type]="discrete_diffusion" [tasks_json]="{\"hellaswag\":3,\"mmlu\":5}")'
    '([hope_name]="eval_niu_1B_fineweb" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/niu_1B_fineweb_edu_100b_potential" [batch_model_type]="causal" [tasks_json]="{\"hellaswag\":3}")'
)


# ==============================================================================
# 3. 脚本固定配置 (通常无需修改)
# ==============================================================================
HOPE_TEMPLATE="train.hope"
MAIN_SCRIPT_PATH="eval.sh"
BASE_OUTPUT_DIR="./evaluation_results"
LOG_FILE="./eval_job_submission.log"
MAX_PARALLEL_JOBS=4


# ==============================================================================
# 4. 任务提交逻辑 (无需修改)
# ==============================================================================

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        log_message "ERROR: $2 file '$1' not found. Please ensure it exists."
        exit 1
    fi
}

# --- 初始化 ---
check_file_exists "$HOPE_TEMPLATE" "Template"
check_file_exists "$MAIN_SCRIPT_PATH" "Main execution script"

echo "--- Eval Job Submission Script Started ---" > "$LOG_FILE"
log_message "INFO: Base output directory: '${BASE_OUTPUT_DIR}'"
log_message "INFO: HOPE template file: '${HOPE_TEMPLATE}'"
log_message "INFO: Main execution script: '${MAIN_SCRIPT_PATH}'"
log_message "INFO: Maximum parallel jobs: ${MAX_PARALLEL_JOBS:-'unlimited'}"

declare -a temp_hope_files=()
declare -a bg_pids=()
declare -a submitted_jobs=()
task_counter=0
total_tasks=${#task_overrides[@]}

# --- 生成并提交任务 ---
log_message "INFO: Generating and submitting ${total_tasks} tasks..."

for override_str in "${task_overrides[@]}"; do
    ((task_counter++))
    log_message "--- Preparing Task ${task_counter}/${total_tasks} ---"
    
    # 1. 合并默认配置和任务特定配置
    unset task_config
    declare -A task_config
    for key in "${!default_config[@]}"; do
        task_config[$key]="${default_config[$key]}"
    done
    
    eval "declare -A current_overrides=${override_str}"
    for key in "${!current_overrides[@]}"; do
        task_config[$key]="${current_overrides[$key]}"
    done

    # 2. 检查并设置核心参数
    if [ -z "${task_config[hope_name]}" ]; then
        log_message "WARNING: Skipping task due to missing 'hope_name' in override string: ${override_str}"
        continue
    fi
    if [ -z "${task_config[target_parent_dir]}" ]; then
        log_message "WARNING: Skipping task due to missing 'target_parent_dir' in override string: ${override_str}"
        continue
    fi
    if [ -z "${task_config[output_subdir]}" ]; then
        task_config[output_subdir]="${task_config[hope_name]}"
    fi

    # 3. 从关联数组中提取所有参数
    hope_name="${task_config[hope_name]}"
    output_subdir="${task_config[output_subdir]}"
    target_parent_dir="${task_config[target_parent_dir]}"
    batch_model_type="${task_config[batch_model_type]}"
    batch_size="${task_config[batch_size]}"
    limit="${task_config[limit]}"
    diffusion_eval_mode="${task_config[diffusion_eval_mode]}"
    tasks_json="${task_config[tasks_json]}"

    # 4. 构建新的 worker 命令
    #    用法: bash eval.sh <OUTPUT_DIR> <TARGET_PARENT_DIR> <BATCH_MODEL_TYPE> \
    #                       <BATCH_SIZE> <LIMIT> <DIFFUSION_EVAL_MODE> <TASKS_JSON>
    full_output_dir="${BASE_OUTPUT_DIR}/${output_subdir}"
    worker_command="bash ${MAIN_SCRIPT_PATH} \
        \"${full_output_dir}\" \
        \"${target_parent_dir}\" \
        \"${batch_model_type}\" \
        \"${batch_size}\" \
        \"${limit}\" \
        \"${diffusion_eval_mode}\" \
        '${tasks_json}'"

    log_message "  Task Name: ${hope_name}"
    log_message "  Output Dir: ${full_output_dir}"
    log_message "  Target Parent Dir: ${target_parent_dir}"
    log_message "  Model Type: ${batch_model_type}"
    log_message "  Batch Size: ${batch_size}"
    log_message "  Limit: ${limit}"
    log_message "  Diffusion Eval Mode: ${diffusion_eval_mode}"
    log_message "  Tasks JSON: ${tasks_json}"
    log_message "  Worker Cmd: (see below)"
    echo "    ${worker_command}" | tee -a "$LOG_FILE"

    # 5. 生成临时 HOPE 文件并提交
    temp_hope_file="${hope_name}.hope"
    
    # [步骤 1] 清理命令中的换行符，将其压缩为一行
    clean_worker_command=$(echo "$worker_command" | tr -s '[:space:]' ' ')

    # [步骤 2] 使用 awk 进行安全替换
    awk -v new_cmd="$clean_worker_command" '
    /^worker.script =/ { 
        print "worker.script = " new_cmd
        next 
    }
    { print }
    ' "${HOPE_TEMPLATE}" > "${temp_hope_file}"

    # [步骤 3] 简单的检查
    if [ ! -s "${temp_hope_file}" ]; then
        log_message "ERROR: Failed to create temporary HOPE file '${temp_hope_file}' (file is empty or awk failed)."
        continue
    fi
    
    # 6. 并行任务控制逻辑
    if [[ "$MAX_PARALLEL_JOBS" -gt 0 && "${#bg_pids[@]}" -ge "$MAX_PARALLEL_JOBS" ]]; then
        log_message "INFO: Reached max parallel jobs ($MAX_PARALLEL_JOBS). Waiting for a job to finish..."
        wait -n
        new_bg_pids=()
        for pid in "${bg_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_bg_pids+=("$pid")
            else
                log_message "INFO: Job with PID $pid has finished."
            fi
        done
        bg_pids=("${new_bg_pids[@]}")
    fi

    log_message "INFO: Submitting: hope run '${temp_hope_file}'"
    hope run "${temp_hope_file}" &
    
    bg_pids+=($!)
    temp_hope_files+=("${temp_hope_file}")
    submitted_jobs+=("${hope_name}:${output_subdir}")
    log_message "INFO: Task '${hope_name}' submitted to background (PID: $!). Temp file: '${temp_hope_file}'"
    log_message "------"
done

# --- 等待并清理 ---
log_message "INFO: ---"
log_message "INFO: All ${#submitted_jobs[@]} task configurations processed. Waiting for ${#bg_pids[@]} background jobs to complete..."

if [ ${#bg_pids[@]} -gt 0 ]; then
    wait
    log_message "INFO: All background 'hope run' processes have finished."
fi

log_message "INFO: Cleaning up ${#temp_hope_files[@]} temporary HOPE files..."
if [ ${#temp_hope_files[@]} -gt 0 ]; then
    if ! rm -f "${temp_hope_files[@]}"; then
        log_message "WARNING: Some temporary HOPE files could not be removed."
    else
        log_message "INFO: Temporary HOPE files cleaned up successfully."
    fi
fi

log_message "--- Eval Job Submission Script Finished ---"
exit 0