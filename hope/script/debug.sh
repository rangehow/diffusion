#!/bin/bash

# ==============================================================================
# 1. DEFINE DEFAULT PARAMETERS
#    All available parameters for the script should have a default value here.
# ==============================================================================
declare -A default_config=(
    [batch_size]=8
    [grad_accum_steps]=8
    [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_large.json"
    [mode]="niu"
    [mlm_prob_start]=1
    [mlm_prob_end]=0.0001
    [mlm_schedule_type]="random"
    [dataset_name]="fineweb_10b"
    [MAX_LENGTH]=1024
    [EPOCHS]=1
    [tail_bias_factor]=1.5
    # --- Add new default parameters here ---
    # [my_new_param]="default_value"
)

# ==============================================================================
# 2. DEFINE THE ORDER OF PARAMETERS FOR THE WORKER SCRIPT
#    This is the SINGLE SOURCE OF TRUTH for the command line argument order.
#    The special key 'output_dir' is handled automatically.
# ==============================================================================
declare -a PARAM_ORDER=(
    "output_dir"            # Special key for the output directory
    "mlm_schedule_type"
    "mlm_prob_start"
    "mlm_prob_end"
    "batch_size"
    "grad_accum_steps"
    "config_path"
    "mode"
    "dataset_name"
    "MAX_LENGTH"
    "EPOCHS"
    "tail_bias_factor"
    # --- Add new parameter keys here in the correct order ---
    # "my_new_param"
)

# ==============================================================================
# 3. DEFINE TASK-SPECIFIC OVERRIDES
#    - 'hope_name' is required.
#    - 'output_subdir' defaults to 'hope_name' if not provided.
# ==============================================================================
task_overrides=(
    '([hope_name]="niu_1B_100b_finefineweb" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=1024 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_1B.json" [mlm_prob_start]=0.99999 [mlm_prob_end]=0.00001)'
    
    '([hope_name]="niu_1B_100b_finefineweb_lowvariance" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=1024 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/modernbert_1B.json" [mlm_prob_start]=0.7 [mlm_prob_end]=0.3)'
    
    '([hope_name]="llada_1B_100b_finefineweb" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=1024 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llada_1b.json" [mlm_prob_start]=0.9 [mlm_prob_end]=0.1)'
    
    '([hope_name]="llama_1B_100b_finefineweb" [dataset_name]="filtered_finefineweb" [MAX_LENGTH]=1024 [config_path]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_config/llama_1B.json" [mlm_prob_start]=0.9 [mlm_prob_end]=0.1)'
    
)

# --- Script Configuration (unchanged) ---
HOPE_TEMPLATE="train.hope"
MAIN_SCRIPT_PATH="train.sh"
BASE_OUTPUT_DIR="diffusion/model_output"
LOG_FILE="./job_submission.log"
MAX_PARALLEL_JOBS=4

# --- Helper Functions (unchanged) ---
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
    # This function is not used in the main loop but is good practice to keep.
    # The 'hope' system or the worker script itself should handle directory creation.
    if [ ! -d "$1" ]; then
        log_message "INFO: Creating output directory: '$1'"
        if ! mkdir -p "$1"; then
            log_message "ERROR: Failed to create directory '$1'."
            exit 1
        fi
    fi
}

# --- Initialization (unchanged) ---
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

# ==============================================================================
# 4. GENERATE AND SUBMIT TASKS (REFACTORED LOGIC)
# ==============================================================================
log_message "INFO: Generating and submitting ${total_tasks} tasks..."

for override_str in "${task_overrides[@]}"; do
    ((task_counter++))
    log_message "--- Preparing Task ${task_counter}/${total_tasks} ---"

    # Step A: Merge default and override configurations for the current task.
    # 'local -A' creates a new associative array for each loop iteration,
    # avoiding the need for 'unset'.
    local -A task_config
    for key in "${!default_config[@]}"; do
        task_config[$key]="${default_config[$key]}"
    done
    
    eval "declare -A current_overrides=${override_str}"
    for key in "${!current_overrides[@]}"; do
        task_config[$key]="${current_overrides[$key]}"
    done

    # Step B: Validate mandatory keys and set defaults
    local hope_name="${task_config[hope_name]}"
    if [ -z "${hope_name}" ]; then
        log_message "WARNING: Skipping task due to missing 'hope_name' in override string: ${override_str}"
        continue
    fi
    # If output_subdir is not specified, use hope_name as the default
    if [ -z "${task_config[output_subdir]}" ]; then
        task_config[output_subdir]="${hope_name}"
    fi

    # Step C: Automatically build the worker command arguments
    # This loop reads the desired order from PARAM_ORDER and builds the arguments array.
    local -a worker_args=()
    for param_key in "${PARAM_ORDER[@]}"; do
        local arg_value=""
        if [[ "$param_key" == "output_dir" ]]; then
            # Special case for the output directory
            arg_value="${BASE_OUTPUT_DIR}/${task_config[output_subdir]}"
        elif [[ -v "task_config[${param_key}]" ]]; then
            # Regular parameter found in our config
            arg_value="${task_config[$param_key]}"
        else
            # Parameter in PARAM_ORDER but not in config (error or empty string)
            log_message "WARNING: Parameter '${param_key}' is in PARAM_ORDER but has no value for task '${hope_name}'. Using empty string."
            arg_value=""
        fi
        worker_args+=("$arg_value")
    done

    # Build the final command safely, with proper quoting
    worker_command="bash ${MAIN_SCRIPT_PATH} \"${worker_args[@]}\""

    log_message "  Task Name: ${hope_name}"
    log_message "  Output Dir: ${BASE_OUTPUT_DIR}/${task_config[output_subdir]}"
    log_message "  Worker Cmd: (see below)"
    echo "    ${worker_command}" | tee -a "$LOG_FILE"

    # Step D: Generate temporary HOPE file and submit (logic unchanged)
    temp_hope_file="${hope_name}.hope"
    escaped_worker_command=$(printf '%s\n' "$worker_command" | sed -e 's/[\/&]/\\&/g')

    if ! sed "s#^worker.script = .*#worker.script = ${escaped_worker_command}#" "${HOPE_TEMPLATE}" > "${temp_hope_file}"; then
        log_message "ERROR: Failed to create temporary HOPE file '${temp_hope_file}'."
        continue
    fi
    
    # Parallel job control (logic unchanged)
    if [[ "$MAX_PARALLEL_JOBS" -gt 0 && "${#bg_pids[@]}" -ge "$MAX_PARALLEL_JOBS" ]]; then
        log_message "INFO: Reached max parallel jobs ($MAX_PARALLEL_JOBS). Waiting..."
        wait -n
        new_bg_pids=()
        for pid in "${bg_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then new_bg_pids+=("$pid"); else log_message "INFO: Job with PID $pid has finished."; fi
        done
        bg_pids=("${new_bg_pids[@]}")
    fi

    log_message "INFO: Submitting: hope run '${temp_hope_file}'"
    # hope run "${temp_hope_file}" &
    
    bg_pids+=($!)
    temp_hope_files+=("${temp_hope_file}")
    submitted_jobs+=("${hope_name}:${task_config[output_subdir]}")
    log_message "INFO: Task '${hope_name}' submitted (PID: $!). Temp file: '${temp_hope_file}'"
    log_message "------"
done

# --- Wait and Cleanup (logic unchanged) ---
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