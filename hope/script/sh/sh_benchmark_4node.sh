#!/bin/bash
########################################################################
# 训练速度基准测试 — 所有 1B 模型在相同 4 节点 (32 GPU) H800 上运行 500 步
#
# 目的：为 rebuttal 提供公平的 wall-clock 训练速度对比
# 对齐变量：
#   - 硬件: 4 × H800 节点 (32 GPUs)
#   - 数据集: sh_filtered_finefineweb
#   - max_length: 2048
#   - effective batch: 4096 seqs/step (bs × ga × 32 GPUs)
#   - max_steps: 500
#   - learning_rate: 2e-4
#   - seed: 42
#   - 从头训练 (不 resume)
#   - save_strategy: "no" (不保存 checkpoint，纯测速)
#
# 不同之处 (由架构决定，无法统一)：
#   - per_device_train_batch_size: BD3LM 只能 bs=2
#   - pad_to_max_length: BD3LM 需要 pad，其他不 pad
#   - grad_accumulation_steps: 根据 bs 调整以保持 4096 seqs/step
########################################################################

cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/hope

# 固定参数
declare -A default_config=(
    [batch_size]=8
    [grad_accum_steps]=16
    [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config/modernbert_1b.json"
    [mode]="niu"
    [mlm_prob_start]=1
    [mlm_prob_end]=0.0001
    [mlm_schedule_type]="cosine_with_min_lr"
    [dataset_name]="sh_filtered_finefineweb"
    [MAX_LENGTH]=2048
    [EPOCHS]=1
    [tail_bias_factor]=100
    [use_daum]=True
    [max_steps]=500
)

task_overrides=(
    #==========================================================================
    # 1. CARD (Ours): bs=8, ga=16 → 8×16×32=4096 seqs/step
    #==========================================================================
    '([hope_name]="benchmark_card" [output_subdir]="benchmark/card" [mode]="niu" [batch_size]=8 [grad_accum_steps]=16 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config/modernbert_1b.json" [tail_bias_factor]=100 [use_daum]=True [mlm_prob_start]=0.99999 [mlm_prob_end]=0.00001)'

    #==========================================================================
    # 2. ARM (Llama): bs=4, ga=32 → 4×32×32=4096 seqs/step
    #==========================================================================
    '([hope_name]="benchmark_arm" [output_subdir]="benchmark/arm" [mode]="llama" [batch_size]=4 [grad_accum_steps]=32 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config/llama_1b.json" [mlm_prob_start]=0.9 [mlm_prob_end]=0.1 [tail_bias_factor]=1.5 [use_daum]=False)'

    #==========================================================================
    # 3. MDLM: bs=8, ga=16 → 8×16×32=4096 seqs/step
    #==========================================================================
    '([hope_name]="benchmark_mdlm" [output_subdir]="benchmark/mdlm" [mode]="mdlm" [batch_size]=8 [grad_accum_steps]=16 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config/llada_1b.json" [mlm_prob_start]=1 [mlm_prob_end]=0.0001 [tail_bias_factor]=1.5 [use_daum]=True)'

    #==========================================================================
    # 4. BD3LM: bs=2, ga=64 → 2×64×32=4096 seqs/step
    #    (BD3LM 内存大，单卡只能 bs=2; 必须 pad_to_max_length)
    #==========================================================================
    '([hope_name]="benchmark_bd3lm" [output_subdir]="benchmark/bd3lm" [mode]="bd3lm" [batch_size]=2 [grad_accum_steps]=64 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_config/bd3lm_1b.json" [mlm_prob_start]=1 [mlm_prob_end]=0.0001 [tail_bias_factor]=1.5 [use_daum]=True)'
)


HOPE_TEMPLATE="hope/benchmark_4node.hope"
MAIN_SCRIPT_PATH="script/sh/benchmark_train.sh"
BASE_OUTPUT_DIR="diffusion/model_output"
LOG_FILE="./benchmark_submission.log"
MAX_PARALLEL_JOBS=1   # 逐个运行，确保不互相抢资源


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
check_file_exists "$MAIN_SCRIPT_PATH" "Main script"

echo "--- Benchmark Submission Script Started ---" > "$LOG_FILE"
log_message "INFO: Base output directory: '${BASE_OUTPUT_DIR}'"
log_message "INFO: HOPE template file: '${HOPE_TEMPLATE}'"
log_message "INFO: Main execution script: '${MAIN_SCRIPT_PATH}'"
log_message "INFO: Jobs will run SEQUENTIALLY (MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS})"

declare -a temp_hope_files=()
declare -a bg_pids=()
declare -a submitted_jobs=()
task_counter=0
total_tasks=${#task_overrides[@]}

# --- 生成并提交任务 ---
log_message "INFO: Generating and submitting ${total_tasks} benchmark tasks..."

for override_str in "${task_overrides[@]}"; do
    ((task_counter++))
    log_message "--- Preparing Benchmark Task ${task_counter}/${total_tasks} ---"
    unset hope_name output_subdir mlm_schedule_type  \
          batch_size grad_accum_steps config_path mode mlm_prob_start mlm_prob_end \
          full_output_dir worker_command max_length EPOCHS tail_bias_factor max_steps
    # 合并配置
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
        log_message "WARNING: Skipping task due to missing 'hope_name'"
        continue
    fi
    if [ -z "${task_config[output_subdir]}" ]; then
        task_config[output_subdir]="${task_config[hope_name]}"
    fi

    # 提取参数
    hope_name="${task_config[hope_name]}"
    output_subdir="${task_config[output_subdir]}"
    batch_size="${task_config[batch_size]}"
    grad_accum_steps="${task_config[grad_accum_steps]}"
    config_path="${task_config[config_path]}"
    mode="${task_config[mode]}"
    mlm_prob_start="${task_config[mlm_prob_start]}"
    mlm_prob_end="${task_config[mlm_prob_end]}"
    mlm_schedule_type="${task_config[mlm_schedule_type]}"
    dataset_name="${task_config[dataset_name]}"
    max_length="${task_config[MAX_LENGTH]}"
    EPOCHS="${task_config[EPOCHS]}"
    tail_bias_factor="${task_config[tail_bias_factor]}"
    use_daum="${task_config[use_daum]}"
    max_steps="${task_config[max_steps]}"

    full_output_dir="${BASE_OUTPUT_DIR}/${output_subdir}"
    
    # 计算 effective batch 用于日志
    effective_batch=$((batch_size * grad_accum_steps * 32))
    
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
        \"${use_daum}\" \
        \"${max_steps}\""

    log_message "  Task Name: ${hope_name}"
    log_message "  Mode: ${mode}"
    log_message "  Output Dir: ${full_output_dir}"
    log_message "  Effective batch: ${effective_batch} seqs/step (bs=${batch_size} × ga=${grad_accum_steps} × 32 GPUs)"
    log_message "  Max steps: ${max_steps}"
    echo "    ${worker_command}" | tee -a "$LOG_FILE"

    # 生成临时 HOPE 文件
    temp_hope_file="${hope_name}.hope"
    escaped_worker_command=$(printf '%s\n' "$worker_command" | sed -e 's/[\/&]/\\&/g')

    if ! sed "s#^worker.script = .*#worker.script = ${escaped_worker_command}#" "${HOPE_TEMPLATE}" > "${temp_hope_file}"; then
        log_message "ERROR: Failed to create temporary HOPE file '${temp_hope_file}'"
        continue
    fi
    
    # 串行：等待上一个完成
    if [[ "$MAX_PARALLEL_JOBS" -gt 0 && "${#bg_pids[@]}" -ge "$MAX_PARALLEL_JOBS" ]]; then
        log_message "INFO: Waiting for previous benchmark to finish..."
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
    hope run "${temp_hope_file}" -Dhope.resource.experiment=fmg_h800_ci -Dmlp.sche.priority=P0 &
    
    bg_pids+=($!)
    temp_hope_files+=("${temp_hope_file}")
    submitted_jobs+=("${hope_name}:${output_subdir}")
    log_message "INFO: Task '${hope_name}' submitted (PID: $!)"
    log_message "------"
done

# --- 等待并清理 ---
log_message "INFO: All ${#submitted_jobs[@]} benchmark tasks submitted. Waiting for completion..."

if [ ${#bg_pids[@]} -gt 0 ]; then
    wait
    log_message "INFO: All benchmark jobs have finished."
fi

log_message "INFO: Cleaning up temporary HOPE files..."
if [ ${#temp_hope_files[@]} -gt 0 ]; then
    rm -f "${temp_hope_files[@]}"
fi

log_message "--- Benchmark Submission Script Finished ---"
log_message ""
log_message "=========================================="
log_message "  结果分析："
log_message "  在每个 benchmark/* 目录下查看 trainer_state.json"
log_message "  提取 log_history 中 step=500 附近的 train_runtime"
log_message "  s/step = train_runtime / 500"
log_message "=========================================="

exit 0
