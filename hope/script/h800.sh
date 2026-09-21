#!/bin/bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/ruanjunhao04/diffusion/hope
# 1. 定义默认参数
declare -A default_config=(
    [batch_size]=8
    [grad_accum_steps]=8
    [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/ruanjunhao04/diffusion/model_config/modernbert_large.json"
    [mode]="niu"
    [mlm_prob_start]=1
    [mlm_prob_end]=0.0001
    [mlm_schedule_type]="linear"
    [dataset_name]="fineweb_10b"
    [MAX_LENGTH]=2048
    [EPOCHS]=1
    [tail_bias_factor]=1.5
    [use_daum]=False
)

# 2. 定义任务特定的覆盖参数
#    现在，你只需要指定 `hope_name`。`output_subdir` 会自动使用 `hope_name` 的值。
#    如果你想让它们不同，仍然可以显式地指定 `[output_subdir]="..."`。
#
#    必填键:
#    - hope_name: HOPE 任务名称，同时也用作默认的输出子目录名。
#
#    可选/常用覆盖键:
#    - output_subdir: (如果想让输出目录名与任务名不同，可以设置此项)
#    - mlm_schedule_type, mask_prob, random_prob, mlm_prob_start, etc.
#
task_overrides=(
    # '([hope_name]="niu_1B_100b_finefineweb" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=1024 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/ruanjunhao04/diffusion/model_config/modernbert_1B.json" [mlm_prob_start]=0.99999 [mlm_prob_end]=0.00001)'
    # '([hope_name]="niu_1B_100b_finefineweb_lowvariance" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=1024 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/ruanjunhao04/diffusion/model_config/modernbert_1B.json" [mlm_prob_start]=0.7 [mlm_prob_end]=0.3)'
    '([hope_name]="guidebench_exp_v2_w_math_improvement" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config/llada_1b.json" [batch_size]=8 [grad_accum_steps]=8 [mlm_prob_start]=0.9 [mlm_prob_end]=0.1 [mode]="llada")'
    # '([hope_name]="arm_1B_100b_2node" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/ruanjunhao04/diffusion/model_config/llama_1b.json" [batch_size]=4 [grad_accum_steps]=16 [mlm_prob_start]=0.9 [mlm_prob_end]=0.1 [mode]="llama")'
    # '([hope_name]="niu_1B_100b_2node" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/ruanjunhao04/diffusion/model_config/modernbert_1b.json" [mlm_prob_start]=0.99999 [mlm_prob_end]=0.00001)'
    # '([hope_name]="niu_1B_100b_lowvariance_2node" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/ruanjunhao04/diffusion/model_config/modernbert_1b.json" [mlm_prob_start]=0.7 [mlm_prob_end]=0.3)'
    # '([hope_name]="niu_1B_tight_2node" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/ruanjunhao04/diffusion/model_config/modernbert_1b.json" [tail_bias_factor]=1 [mlm_prob_start]=0.99999 [mlm_prob_end]=0.00001)'
    # '([hope_name]="niu_1B_100b_fast_2node" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_config/modernbert_1b_swa.json" [mlm_prob_start]=0.99999 [mlm_prob_end]=0.00001)'
    # '([hope_name]="niu_1B_100b_daum_2node" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/ruanjunhao04/diffusion/model_config/modernbert_1b.json" [use_daum]=True [mlm_prob_start]=0.99999 [mlm_prob_end]=0.00001)'
    # '([hope_name]="niu_400M_finefineweb" [dataset_name]="filtered_finefineweb" [batch_size]=16 [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_config/modernbert_large.json" [mlm_prob_start]=0.99999 [mlm_prob_end]=0.00001)'
    # '([hope_name]="niu_400M_finefineweb_lowvariance" [dataset_name]="filtered_finefineweb" [batch_size]=16 [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_config/modernbert_large.json" [mlm_prob_start]=0.7 [mlm_prob_end]=0.3)'
    # '([hope_name]="llada_400M_finefineweb" [dataset_name]="filtered_finefineweb" [batch_size]=8 [grad_accum_steps]=16 [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_config/llada_large.json" [mlm_prob_start]=0.9 [mlm_prob_end]=0.1 [mode]="llada")'
    # '([hope_name]="llama_400M_finefineweb" [dataset_name]="filtered_finefineweb" [batch_size]=8 [grad_accum_steps]=16 [MAX_LENGTH]=2048 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/diffusion/model_config/llama_400M.json" [mlm_prob_start]=0.9 [mlm_prob_end]=0.1 [mode]="llama")'
)


HOPE_TEMPLATE="train_8node.hope"
MAIN_SCRIPT_PATH="train_2node.sh"
BASE_OUTPUT_DIR="diffusion/model_output"
LOG_FILE="./job_submission.log"
MAX_PARALLEL_JOBS=4


log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        log_message "ERROR: $2 file '$1' not found. Please ensure it exists."
        exit 1
    fi
}

ensure_dir_exists() {
    if [ ! -d "$1" ]; then
        log_message "INFO: Creating output directory: '$1'"
        if ! mkdir -p "$1"; then
            log_message "ERROR: Failed to create directory '$1'."
            exit 1
        fi
    fi
}


# --- 初始化 ---
check_file_exists "$HOPE_TEMPLATE" "Template"
check_file_exists "$MAIN_SCRIPT_PATH" "Main script"

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
    unset hope_name output_subdir mlm_schedule_type  \
          batch_size grad_accum_steps config_path mode mlm_prob_start mlm_prob_end \
          full_output_dir worker_command max_length EPOCHS tail_bias_factor
    # 1. 合并配置
    unset task_config
    declare -A task_config
    for key in "${!default_config[@]}"; do
        task_config[$key]="${default_config[$key]}"
    done
    
    eval "declare -A current_overrides=${override_str}"
    for key in "${!current_overrides[@]}"; do
        task_config[$key]="${current_overrides[$key]}"
    done


    if [ -z "${task_config[hope_name]}" ]; then
        log_message "WARNING: Skipping task due to missing 'hope_name' in override string: ${override_str}"
        continue
    fi
    # 如果 output_subdir 未指定，则使用 hope_name 作为默认值
    if [ -z "${task_config[output_subdir]}" ]; then
        task_config[output_subdir]="${task_config[hope_name]}"
    fi

    # 3. 从关联数组中提取所有参数
    hope_name="${task_config[hope_name]}"
    output_subdir="${task_config[output_subdir]}"
    batch_size="${task_config[batch_size]}"
    grad_accum_steps="${task_config[grad_accum_steps]}"
    config_path="${task_config[config_path]}"
    mode="${task_config[mode]}"
    mlm_prob_start="${task_config[mlm_prob_start]}"
    mlm_prob_end="${task_config[mlm_prob_end]}"
    dataset_name="${task_config[dataset_name]}"
    max_length="${task_config[MAX_LENGTH]}"
    EPOCHS="${task_config[EPOCHS]}"
    tail_bias_factor="${task_config[tail_bias_factor]}"
    use_daum="${task_config[use_daum]}"
    # 4. 构建 worker 命令 (无需改动)
    #    用法: bash train.sh <OUTPUT_DIR> <MLM_SCHEDULE_TYPE> <MASK_PROB> <RANDOM_PROB> <MLM_PROB_START> <MLM_PROB_END> <BATCH_SIZE> <GRAD_ACCUM_STEPS> <CONFIG_PATH> <MODE>
    full_output_dir="${BASE_OUTPUT_DIR}/${output_subdir}"
    worker_command="bash ${MAIN_SCRIPT_PATH} \
        \"${full_output_dir}\" \
        \"${mlm_schedule_type}\" \
        \"${mlm_prob_start}\" \
        \"${mlm_prob_end}\" \
        \"${batch_size}\" \
        \"${grad_accum_steps}\" \
        \"${config_path}\" \
        \"${mode}\" \
        \"${dataset_name}\" \
        \"${max_length}\" \
        \"${EPOCHS}\" \
        \"${tail_bias_factor}\" \
        \"${use_daum}\""

    log_message "  Task Name: ${hope_name}"
    log_message "  Output Dir: ${full_output_dir}"
    log_message "  Worker Cmd: (see below)"
    echo "    ${worker_command}" | tee -a "$LOG_FILE" # 使用 echo 以便正确处理多行命令的缩进

    # 5. 生成临时 HOPE 文件并提交 (逻辑不变)
    temp_hope_file="${hope_name}.hope"
    escaped_worker_command=$(printf '%s\n' "$worker_command" | sed -e 's/[\/&]/\\&/g')

    if ! sed "s#^worker.script = .*#worker.script = ${escaped_worker_command}#" "${HOPE_TEMPLATE}" > "${temp_hope_file}"; then
        log_message "ERROR: Failed to create temporary HOPE file '${temp_hope_file}' from template '${HOPE_TEMPLATE}'."
        continue
    fi
    
    # 并行任务控制逻辑
    if [[ "$MAX_PARALLEL_JOBS" -gt 0 && "${#bg_pids[@]}" -ge "$MAX_PARALLEL_JOBS" ]]; then
        log_message "INFO: Reached max parallel jobs ($MAX_PARALLEL_JOBS). Waiting for a job to finish..."
        wait -n
        # 清理已完成的进程ID
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
    # hope run "${temp_hope_file}" -Dhope.resource.experiment=fmg_h800_ci -Dmlp.sche.priority=P0  &
    hope run "${temp_hope_file}" &
    # hope run "${temp_hope_file}"&
    bg_pids+=($!)
    temp_hope_files+=("${temp_hope_file}")
    submitted_jobs+=("${hope_name}:${output_subdir}")
    log_message "INFO: Task '${hope_name}' submitted to background (PID: $!). Temp file: '${temp_hope_file}'"
    log_message "------"
done

# --- 等待并清理 (逻辑不变) ---
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