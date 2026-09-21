#!/bin/bash
# hope_lm1b.sh
# ==============================================================================
# 1. 定义默认参数
# ==============================================================================
declare -A default_config=(
    [mode]="niu"
    [config_file]="modernbert_110m.json"
    [dataset_name]="lm1b_train"
    [epochs]=1
    [batch_size]=64
    [max_steps]=1000000
    [gradient_accumulation_steps]=1
    [ema_decay]=0.9999
    [lr_scheduler_kwargs]="{}"
    [lr_scheduler_type]="constant_with_warmup"
    # (修改 1/4) 新增 warmup 参数的默认值
    [warmup_steps]=2500
    [warmup_ratios]=0.0
)

# ==============================================================================
# 2. 定义任务特定的覆盖参数
# ==============================================================================
task_overrides=(
    # '([hope_name]="mdlm_110m_lm1b_ema" [mode]="mdlm" [config_file]="llada_110m.json")'
    # '([hope_name]="arm_110m_lm1b_ema" [mode]="llama" [config_file]="llama_110m.json")'
    # '([hope_name]="niu_110m_lm1b_ntp" [config_file]="modernbert_110m.json" )'
    # '([hope_name]="llada_110m_lm1b_ema" [mode]="llada" [config_file]="llada_110m.json" )'
    '([hope_name]="bd3lm_110m_lm1b_ema_fixed_1217_2node" [batch_size]=32 [mode]="bd3lm" [config_file]="bd3lm_110m.json")'
)


# ==============================================================================
# 3. 脚本固定配置 (通常无需修改)
# ==============================================================================
HOPE_TEMPLATE="train_2node.hope"
# HOPE_TEMPLATE="train.hope"
MAIN_SCRIPT_PATH="lm1b.sh"
BASE_OUTPUT_DIR="diffusion/model_output"
LOG_FILE="./job_submission.log"
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

echo "--- Job Submission Script Started ---" > "$LOG_FILE"
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
    if [ -z "${task_config[output_subdir]}" ]; then
        task_config[output_subdir]="${task_config[hope_name]}"
    fi

    # 3. 从关联数组中提取所有参数
    hope_name="${task_config[hope_name]}"
    output_subdir="${task_config[output_subdir]}"
    mode="${task_config[mode]}"
    config_file="${task_config[config_file]}"
    dataset_name="${task_config[dataset_name]}"
    epochs="${task_config[epochs]}"
    batch_size="${task_config[batch_size]}"
    max_steps="${task_config[max_steps]}"
    gradient_accumulation_steps="${task_config[gradient_accumulation_steps]}"
    ema_decay="${task_config[ema_decay]}"
    lr_scheduler_kwargs="${task_config[lr_scheduler_kwargs]}"
    lr_scheduler_type="${task_config[lr_scheduler_type]}"
    # (修改 2/4) 提取 warmup 参数
    warmup_steps="${task_config[warmup_steps]}"
    warmup_ratios="${task_config[warmup_ratios]}"

    # 4. 构建新的 worker 命令
    #    (修改 3/4) 更新用法说明，增加第12和13个参数
    #    用法: bash lm1b.sh <...> <LR_SCHEDULER_TYPE> <WARMUP_STEPS> <WARMUP_RATIOS>
    full_output_dir="${BASE_OUTPUT_DIR}/${output_subdir}"
    worker_command="bash ${MAIN_SCRIPT_PATH} \
        \"${full_output_dir}\" \
        \"${mode}\" \
        \"${config_file}\" \
        \"${dataset_name}\" \
        \"${epochs}\" \
        \"${batch_size}\" \
        \"${max_steps}\" \
        \"${gradient_accumulation_steps}\" \
        \"${ema_decay}\" \
        \"${lr_scheduler_kwargs}\" \
        \"${lr_scheduler_type}\" \
        \"${warmup_steps}\" \
        \"${warmup_ratios}\"" # 将第12, 13个参数添加到命令中

    log_message "  Task Name: ${hope_name}"
    log_message "  Output Dir: ${full_output_dir}"
    log_message "  Max Steps: ${max_steps}"
    log_message "  EMA Decay: ${ema_decay}"
    log_message "  LR Scheduler Kwargs: ${lr_scheduler_kwargs}"
    log_message "  LR Scheduler Type: ${lr_scheduler_type}"
    # (修改 4/4) 新增日志，方便调试
    log_message "  Warmup Steps: ${warmup_steps}"
    log_message "  Warmup Ratios: ${warmup_ratios}"
    log_message "  Worker Cmd: (see below)"
    echo "    ${worker_command}" | tee -a "$LOG_FILE"

    # 5. 生成临时 HOPE 文件并提交
    # ... (这是原来的第 5 步位置) ...

    # 5. 生成临时 HOPE 文件并提交
    # 5. 生成临时 HOPE 文件并提交
    temp_hope_file="${hope_name}.hope"
    
    # [步骤 1] 清理命令中的换行符，将其压缩为一行
    clean_worker_command=$(echo "$worker_command" | tr -s '[:space:]' ' ')

    # [步骤 2] 使用 awk 进行安全替换 (核心修改)
    # 说明: 
    # -v new_cmd="..." : 把 shell 变量安全地传给 awk 变量 new_cmd
    # /^worker.script =/ : 匹配以 worker.script = 开头的行
    # print "..." : 打印新的行
    # next : 跳过当前行的默认打印，避免重复
    # { print } : 对于其他所有行，原样打印
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

log_message "--- Job Submission Script Finished ---"
exit 0