#!/bin/bash

# 1. 定义默认参数
declare -A default_config=(
    [batch_size]=8
    [max_length]=2048
    [max_eval_samples]=""
    [use_ema]="True"
)

# 2. 定义要评估的数据集列表
DATASETS=(
    "sh_lm1b_test"
    "sh_ag_news"
    "sh_ptb"
    "sh_lambada"
    "sh_pubmed"
    "sh_arxiv"
    "sh_openwebtext"
    "sh_wikitext"  # 按需添加更多数据集
)

# 3. 定义模型配置 (不包含 dataset_name 和 output_subdir)
model_configs=(
    # '([model_name]="bd3lm" [checkpoint_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/bd3lm_main_exp/checkpoint-77335" [mode]="bd3lm")'
    # '([model_name]="niu" [checkpoint_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_main_exp/checkpoint-77335" [mode]="niu")'
    '([model_name]="arm" [checkpoint_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/arm_main_exp" [mode]="llama")'
    '([model_name]="mdlm" [checkpoint_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/mdlm_main_exp/checkpoint-77335" [mode]="mdlm")'
)

# 4. 动态生成 task_overrides (模型 × 数据集 的笛卡尔积)
task_overrides=()
for model_config in "${model_configs[@]}"; do
    eval "declare -A mc=${model_config}"
    for dataset in "${DATASETS[@]}"; do
        # 生成简短的数据集标识用于命名
        dataset_short="${dataset#sh_}"  # 移除 "sh_" 前缀
        
        task_overrides+=(
            "([hope_name]=\"eval_${mc[model_name]}_${dataset_short}\" \
              [checkpoint_path]=\"${mc[checkpoint_path]}\" \
              [mode]=\"${mc[mode]}\" \
              [dataset_name]=\"${dataset}\" \
              [output_subdir]=\"${mc[model_name]}_${dataset_short}\")"
        )
    done
    unset mc
done

# 使用较小的 hope 模板
HOPE_TEMPLATE="../../hope/sh_fmg_1node.hope"
MAIN_SCRIPT_PATH="eval_loss.sh"
BASE_OUTPUT_DIR="diffusion/eval_output"
MAX_PARALLEL_JOBS=4

# 仅输出到终端
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        log_message "ERROR: $2 file '$1' not found. Please ensure it exists."
        exit 1
    fi
}

# --- 初始化 ---
check_file_exists "$HOPE_TEMPLATE" "Template"
check_file_exists "$MAIN_SCRIPT_PATH" "Main script"

echo "--- Eval Job Submission Script Started ---"
log_message "INFO: Base output directory: '${BASE_OUTPUT_DIR}'"
log_message "INFO: HOPE template file: '${HOPE_TEMPLATE}'"
log_message "INFO: Main execution script: '${MAIN_SCRIPT_PATH}'"
log_message "INFO: Models to evaluate: ${#model_configs[@]}"
log_message "INFO: Datasets to evaluate: ${DATASETS[*]}"

declare -a temp_hope_files=()
declare -a bg_pids=()
declare -a submitted_jobs=()
task_counter=0
total_tasks=${#task_overrides[@]}

# --- 生成并提交任务 ---
log_message "INFO: Generating and submitting ${total_tasks} evaluation tasks (${#model_configs[@]} models × ${#DATASETS[@]} datasets)..."

for override_str in "${task_overrides[@]}"; do
    ((task_counter++))
    log_message "--- Preparing Eval Task ${task_counter}/${total_tasks} ---"
    
    # 重置变量
    unset hope_name checkpoint_path mode dataset_name max_length batch_size output_subdir max_eval_samples use_ema
    
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

    # 验证必要参数
    if [ -z "${task_config[hope_name]}" ]; then
        log_message "WARNING: Skipping task due to missing 'hope_name'"
        continue
    fi
    if [ -z "${task_config[checkpoint_path]}" ]; then
        log_message "WARNING: Skipping task due to missing 'checkpoint_path'"
        continue
    fi

    # 如果 output_subdir 未指定，则使用 hope_name 作为默认值
    if [ -z "${task_config[output_subdir]}" ]; then
        task_config[output_subdir]="${task_config[hope_name]}"
    fi

    # 3. 提取参数
    hope_name="${task_config[hope_name]}"
    checkpoint_path="${task_config[checkpoint_path]}"
    mode="${task_config[mode]}"
    dataset_name="${task_config[dataset_name]}"
    max_length="${task_config[max_length]}"
    batch_size="${task_config[batch_size]}"
    output_subdir="${task_config[output_subdir]}"
    max_eval_samples="${task_config[max_eval_samples]}"
    use_ema="${task_config[use_ema]}"

    full_output_dir="${BASE_OUTPUT_DIR}/${output_subdir}"

    # 4. 构建 worker 命令
    worker_command="bash ${MAIN_SCRIPT_PATH} \
        \"${checkpoint_path}\" \
        \"${mode}\" \
        \"${dataset_name}\" \
        \"${max_length}\" \
        \"${batch_size}\" \
        \"${full_output_dir}\" \
        \"${max_eval_samples}\" \
        \"${use_ema}\""

    log_message "  Task Name: ${hope_name}"
    log_message "  Checkpoint: ${checkpoint_path}"
    log_message "  Mode: ${mode}"
    log_message "  Dataset: ${dataset_name}"
    log_message "  Use EMA: ${use_ema}"
    log_message "  Output Dir: ${full_output_dir}"
    echo "  Worker Cmd: ${worker_command}"

    # 5. 生成临时 HOPE 文件并提交
    temp_hope_file="${hope_name}.hope"
    escaped_worker_command=$(printf '%s\n' "$worker_command" | sed -e 's/[\/&]/\\&/g')

    if ! sed "s#^worker.script = .*#worker.script = ${escaped_worker_command}#" "${HOPE_TEMPLATE}" > "${temp_hope_file}"; then
        log_message "ERROR: Failed to create temporary HOPE file '${temp_hope_file}'"
        continue
    fi
    
    # 并行任务控制
    if [[ "$MAX_PARALLEL_JOBS" -gt 0 && "${#bg_pids[@]}" -ge "$MAX_PARALLEL_JOBS" ]]; then
        log_message "INFO: Reached max parallel jobs ($MAX_PARALLEL_JOBS). Waiting..."
        wait -n
        new_bg_pids=()
        for pid in "${bg_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_bg_pids+=("$pid")
            fi
        done
        bg_pids=("${new_bg_pids[@]}")
    fi

    log_message "INFO: Submitting: hope run '${temp_hope_file}'"
    hope run "${temp_hope_file}" -Dhope.resource.experiment=fmg_h800_ci -Dmlp.sche.priority=P0 &
    bg_pids+=($!)
    temp_hope_files+=("${temp_hope_file}")
    submitted_jobs+=("${hope_name}")
    log_message "INFO: Task '${hope_name}' submitted (PID: $!)"
    log_message "------"
done

# --- 等待并清理 ---
log_message "INFO: All ${#submitted_jobs[@]} tasks submitted. Waiting for completion..."

if [ ${#bg_pids[@]} -gt 0 ]; then
    wait
    log_message "INFO: All background processes finished."
fi

log_message "INFO: Cleaning up temporary HOPE files..."
if [ ${#temp_hope_files[@]} -gt 0 ]; then
    rm -f "${temp_hope_files[@]}" 2>/dev/null
    log_message "INFO: Cleanup complete."
fi

log_message "--- Eval Job Submission Script Finished ---"
exit 0