#!/bin/bash
# hope_eval_v2.sh
# ==============================================================================
# HOPE Evaluation Job Submission Script - V2 with Parallelization Support
# ==============================================================================
# Features:
# - Split checkpoints across multiple jobs (checkpoints_per_job)
# - Split tasks across multiple jobs (split_tasks_into_jobs)
# - Automatic checkpoint discovery
# - Configurable parallelism
# ==============================================================================

set -e

# ==============================================================================
# 1. Global Parallelization Settings
# ==============================================================================

# How many checkpoints should each job process?
# Set to 0 or empty to process all checkpoints in one job (original behavior)
CHECKPOINTS_PER_JOB=10

# Should tasks be split into separate jobs?
# Options: "none" (all tasks in one job), "each" (one job per task), or a number (tasks per job)
SPLIT_TASKS_MODE="none"

# Maximum parallel hope submissions (to avoid overwhelming the scheduler)
MAX_PARALLEL_SUBMISSIONS=20

# ==============================================================================
# 2. Default Parameters (inherited by all task configurations)
# ==============================================================================

declare -A default_config=(
    [batch_model_type]="discrete_diffusion"
    [batch_size]=16
    [limit]=0
    [diffusion_eval_mode]="mc"
    [diffusion_type]="mdlm"
    [mc_num]=32
)

# ==============================================================================
# 3. Task Configurations
# ==============================================================================
# Each entry defines an evaluation target. Checkpoints are auto-discovered.
#
# Required fields:
#   - hope_name: Base name for the job (will be suffixed with batch/task info)
#   - target_parent_dir: Directory containing checkpoints OR a single model
#
# Optional overrides:
#   - batch_model_type, batch_size, limit, diffusion_eval_mode, etc.
#   - tasks_json: JSON object mapping task names to few-shot counts
#   - checkpoints_per_job: Override global setting for this config
#   - split_tasks_mode: Override global setting for this config
#

task_overrides=(
    # arm main
    # '([hope_name]="llm_platform_consumer_submit-2026-01-13-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/arm_main_exp" [batch_model_type]="causal" [tasks_json]="{\"hellaswag\":3,\"mmlu_redux_corrected \":5,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"sciq\":0}")'

    '([hope_name]="llm_platform_consumer_submit-2026-01-25-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/arm_main_exp" [batch_model_type]="causal" [tasks_json]="{\"mmlu_redux_corrected \":5}")'

    # niu main
    # '([hope_name]="llm_platform_consumer_submit-2026-01-14-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_main_exp" [batch_model_type]="causal"  [tasks_json]="{\"hellaswag\":3,\"mmlu_redux_corrected \":5,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"sciq\":0}")'

    # mdlm main
    # '([hope_name]="llm_platform_consumer_submit-2026-01-14-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/mdlm_main_exp" [batch_model_type]="discrete_diffusion" [diffusion_type]="mdlm" [tasks_json]="{\"hellaswag\":3,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"mmlu_redux_corrected\":5,\"sciq\":0}")'

    # bd3lm main
    # '([hope_name]="llm_platform_consumer_submit-2026-01-15-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/bd3lm_main_exp" [batch_model_type]="discrete_diffusion" [batch_size]=2 [diffusion_type]="bd3lm" [tasks_json]="{\"hellaswag\":3,\"mmlu\":5,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"sciq\":0}")'
    


    # '([hope_name]="llm_platform_consumer_submit-2026-01-14-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_tight_2node/checkpoint-309339" [batch_model_type]="causal"  [tasks_json]="{\"hellaswag\":3,\"mmlu_redux_corrected \":5,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"sciq\":0}")'

    # '([hope_name]="llm_platform_consumer_submit-2026-01-14-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_loose_2node/checkpoint-309339" [batch_model_type]="causal"  [tasks_json]="{\"hellaswag\":3,\"mmlu\":5,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"sciq\":0}")'

    # '([hope_name]="llm_platform_consumer_submit-2026-01-14-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/old_niu_1b_100b_2node/checkpoint-309339" [batch_model_type]="causal"  [tasks_json]="{\"hellaswag\":3,\"mmlu\":5,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"sciq\":0}")'

    # '([hope_name]="llm_platform_consumer_submit-2026-01-14-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_100b_daum_2node/checkpoint-309339" [batch_model_type]="causal"  [tasks_json]="{\"hellaswag\":3,\"mmlu\":5,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"sciq\":0}")'


    # '([hope_name]="llm_platform_consumer_submit-2026-01-14-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_main_exp" [batch_model_type]="causal" [batch_size]=4 [tasks_json]="{\"mmlu\":5}")'
    # mdlm main
    # '([hope_name]="llm_platform_consumer_submit-2026-01-14-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/mdlm_main_exp" [batch_model_type]="discrete_diffusion" [diffusion_type]="mdlm" [tasks_json]="{\"hellaswag\":3,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"mmlu_redux_corrected\":5,\"sciq\":0}")'

    # '([hope_name]="llm_platform_consumer_submit-2026-01-14-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/mdlm_main_exp" [batch_model_type]="discrete_diffusion" [diffusion_type]="mdlm" [batch_size]=4 [tasks_json]="{\"mmlu_redux_corrected\":5}")'

    # '([hope_name]="llm_platform_consumer_submit-2026-01-15-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/bd3lm_main_exp" [batch_model_type]="discrete_diffusion" [batch_size]=2 [diffusion_type]="bd3lm" [tasks_json]="{\"hellaswag\":3,\"mmlu\":5,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"sciq\":0}")'

    # '([hope_name]="llm_platform_consumer_submit-2026-01-15-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/bd3lm_main_exp" [batch_model_type]="discrete_diffusion" [batch_size]=2 [diffusion_type]="bd3lm" [tasks_json]="{\"sciq\":0}")'

    # arm potential
    # '([hope_name]="llm_platform_consumer_submit-2026-01-16-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/arm_1B_fineweb_edu_100b_potential" [batch_model_type]="causal" [tasks_json]="{\"hellaswag\":3}")'

    # mdlm potential
    # '([hope_name]="llm_platform_consumer_submit-2026-01-17-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/mdlm_1B_fineweb_edu_100b_potential" [batch_model_type]="discrete_diffusion" [diffusion_type]="mdlm" [tasks_json]="{\"hellaswag\":3}")'

    # niu potential
    # '([hope_name]="llm_platform_consumer_submit-2026-01-18-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/niu_1B_fineweb_edu_100b_potential" [batch_model_type]="causal" [tasks_json]="{\"hellaswag\":3}")'
    
    # bd3lm potential
    # '([hope_name]="llm_platform_consumer_submit-2026-01-19-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/bd3lm_1B_fineweb_edu_100b_potential" [batch_model_type]="discrete_diffusion" [batch_size]=2 [diffusion_type]="bd3lm" [tasks_json]="{\"hellaswag\":3}")'

    # '([hope_name]="llm_platform_consumer_submit-2026-01-19-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/bd3lm_1B_fineweb_edu_100b_potential/checkpoint-6636" [batch_model_type]="discrete_diffusion" [batch_size]=2 [diffusion_type]="bd3lm" [tasks_json]="{\"hellaswag\":3}")'
    # '([hope_name]="llm_platform_consumer_submit-2026-01-19-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/bd3lm_1B_fineweb_edu_100b_potential/checkpoint-7584" [batch_model_type]="discrete_diffusion" [batch_size]=2 [diffusion_type]="bd3lm" [tasks_json]="{\"hellaswag\":3}")'
    # '([hope_name]="llm_platform_consumer_submit-2026-01-19-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/bd3lm_1B_fineweb_edu_100b_potential/checkpoint-8532" [batch_model_type]="discrete_diffusion" [batch_size]=2 [diffusion_type]="bd3lm" [tasks_json]="{\"hellaswag\":3}")'
    # '([hope_name]="llm_platform_consumer_submit-2026-01-19-10-15-41" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/model_output/bd3lm_1B_fineweb_edu_100b_potential/checkpoint-9480" [batch_model_type]="discrete_diffusion" [batch_size]=2 [diffusion_type]="bd3lm" [tasks_json]="{\"hellaswag\":3}")'
    
    # '([hope_name]="eval_niu_1B_fineweb" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/niu_1B_fineweb_edu_100b_potential" [batch_model_type]="causal" [tasks_json]="{\"hellaswag\":3,\"mmlu\":5,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"sciq\":0}")'
    # '([hope_name]="eval_niu_1B_fineweb" [target_parent_dir]="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/diffusion/model_output/niu_1B_fineweb_edu_100b_potential" [batch_model_type]="causal" [tasks_json]="{\"hellaswag\":3,\"mmlu\":5,\"arc_easy\":25,\"arc_challenge\":25,\"piqa\":0,\"winogrande\":5,\"commonsense_qa\":7,\"truthfulqa_mc2\":0,\"sciq\":0}")'
)

# ["mmlu"]=5
# ["arc_easy"]=25
# ["arc_challenge"]=25
# ["piqa"]=0
# ["winogrande"]=5
# ["commonsense_qa"]=7
# ["truthfulqa_mc2"]=0
# ["sciq"]=0

# ==============================================================================
# 4. Fixed Configuration (usually no need to modify)
# ==============================================================================

HOPE_TEMPLATE="../../hope/sh_fmg_1node.hope"
MAIN_SCRIPT_PATH="eval_sh.sh"
BASE_OUTPUT_DIR="./evaluation_results"

# Global variable to store the last created temp hope file (used to avoid command substitution)
LAST_TEMP_HOPE_FILE=""

# ==============================================================================
# 5. Helper Functions
# ==============================================================================

log_message() {
    # Only echo to stdout, do not write to file
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        log_message "ERROR: $2 file '$1' not found."
        exit 1
    fi
}

# Discover checkpoints in a directory
# Returns: newline-separated list of checkpoint paths
discover_checkpoints() {
    local target_dir="$1"
    local checkpoints=()

    if [ ! -d "$target_dir" ]; then
        echo ""
        return
    fi

    # Look for checkpoint-* directories
    while IFS= read -r cp_path; do
        checkpoints+=("$cp_path")
    done < <(find "$target_dir" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V)

    # If no checkpoints found, check if target_dir itself is a model
    if [ ${#checkpoints[@]} -eq 0 ]; then
        if [ -f "${target_dir}/config.json" ]; then
            checkpoints+=("$target_dir")
        fi
    fi

    printf '%s\n' "${checkpoints[@]}"
}

# Split array into chunks
# Usage: split_array
# Output: Sets CHUNKS as an array of space-separated chunk strings
split_into_chunks() {
    local chunk_size=$1
    shift
    local arr=("$@")
    local total=${#arr[@]}

    CHUNKS=()

    if [ "$chunk_size" -le 0 ] || [ "$chunk_size" -ge "$total" ]; then
        # No splitting needed - return all as one chunk
        CHUNKS+=("${arr[*]}")
        return
    fi

    local i=0
    while [ $i -lt $total ]; do
        local chunk_arr=()
        local j=0
        while [ $j -lt $chunk_size ] && [ $((i + j)) -lt $total ]; do
            chunk_arr+=("${arr[$((i + j))]}")
            ((j++)) || true
        done
        CHUNKS+=("${chunk_arr[*]}")
        ((i += chunk_size)) || true
    done
}

# Parse tasks JSON into arrays
# Sets: TASK_NAMES_ARR and TASK_SHOTS_ARR
parse_tasks_json() {
    local tasks_json="$1"
    eval "$(python3 -c "
import json, sys, shlex
try:
    tasks = json.loads('${tasks_json}')
    names = []
    shots = []
    for task, shot in tasks.items():
        names.append(shlex.quote(str(task)))
        shots.append(str(shot))
    print(f'TASK_NAMES_ARR=({\" \".join(names)})')
    print(f'TASK_SHOTS_ARR=({\" \".join(shots)})')
except Exception as e:
    print(f'echo \"Error parsing JSON: {e}\"', file=sys.stderr)
    sys.exit(1)
")"
}

# Convert task arrays back to JSON
# Usage: tasks_to_json
tasks_to_json() {
    local -n names_ref=$1
    local -n shots_ref=$2
    python3 -c "
import json
names = '''${names_ref[*]}'''.split()
shots = '''${shots_ref[*]}'''.split()
result = {n: int(s) for n, s in zip(names, shots)}
print(json.dumps(result))
"
}

# Generate and submit a single hope job
# Arguments: hope_name, output_dir, checkpoint_list, model_type, batch_size, limit, eval_mode, tasks_json
# Sets: LAST_TEMP_HOPE_FILE with the created hope file path
submit_hope_job() {
    local job_name="$1"
    local output_dir="$2"
    local checkpoint_list="$3"  # Space-separated list of checkpoint paths
    local model_type="$4"
    local batch_size="$5"
    local limit="$6"
    local eval_mode="$7"
    local tasks_json="$8"
    local diffusion_type="${9:-mdlm}"
    local mc_num="${10:-128}"

    # Create temporary hope file
    local temp_hope_file="${job_name}.hope"

    # Build worker command - note we pass checkpoint list directly
    local worker_command="bash ${MAIN_SCRIPT_PATH} \"${output_dir}\" \"${checkpoint_list}\" \"${model_type}\" \"${batch_size}\" \"${limit}\" \"${eval_mode}\" '${tasks_json}' \"${diffusion_type}\" \"${mc_num}\""

    # Clean command (remove extra whitespace)
    local clean_worker_command=$(echo "$worker_command" | tr -s '[:space:]' ' ')

    # Generate hope file from template
    awk -v new_cmd="$clean_worker_command" '
    /^worker.script =/ {
        print "worker.script = " new_cmd
        next
    }
    { print }
    ' "${HOPE_TEMPLATE}" > "${temp_hope_file}"

    if [ ! -s "${temp_hope_file}" ]; then
        log_message "ERROR: Failed to create hope file '${temp_hope_file}'"
        LAST_TEMP_HOPE_FILE=""
        return 1
    fi

    log_message "INFO: Submitting job: ${job_name}"

    # Run hope - output goes directly to terminal (no command substitution)
    hope run "${temp_hope_file}" -Dhope.resource.experiment=fmg_h800_ci -Dmlp.sche.priority=P0 &

    # Set the global variable instead of echoing
    LAST_TEMP_HOPE_FILE="${temp_hope_file}"
}

# ==============================================================================
# 6. Main Execution Logic
# ==============================================================================

main() {
    # Initialize
    check_file_exists "$HOPE_TEMPLATE" "Template"
    check_file_exists "$MAIN_SCRIPT_PATH" "Main execution script"

    log_message "--- Eval Job Submission Script V2 Started ---"
    log_message "INFO: Checkpoint splitting: ${CHECKPOINTS_PER_JOB:-all} per job"
    log_message "INFO: Task splitting mode: ${SPLIT_TASKS_MODE}"
    log_message "INFO: Max parallel submissions: ${MAX_PARALLEL_SUBMISSIONS}"

    declare -a temp_hope_files=()
    declare -a bg_pids=()
    local total_jobs_submitted=0

    # Process each task configuration
    for override_str in "${task_overrides[@]}"; do
        log_message "========================================"

        # Merge default config with overrides
        unset task_config
        declare -A task_config
        for key in "${!default_config[@]}"; do
            task_config[$key]="${default_config[$key]}"
        done

        eval "declare -A current_overrides=${override_str}"
        for key in "${!current_overrides[@]}"; do
            task_config[$key]="${current_overrides[$key]}"
        done

        # Validate required fields
        if [ -z "${task_config[hope_name]}" ] || [ -z "${task_config[target_parent_dir]}" ]; then
            log_message "WARNING: Skipping config - missing hope_name or target_parent_dir"
            continue
        fi

        local base_hope_name="${task_config[hope_name]}"
        local target_dir="${task_config[target_parent_dir]}"
        local tasks_json="${task_config[tasks_json]:-'{\"hellaswag\":3}'}"

        # Use config-specific settings or fall back to global
        local ckpt_per_job="${task_config[checkpoints_per_job]:-$CHECKPOINTS_PER_JOB}"
        local split_tasks="${task_config[split_tasks_mode]:-$SPLIT_TASKS_MODE}"

        log_message "INFO: Processing config: ${base_hope_name}"
        log_message "INFO: Target directory: ${target_dir}"

        # Discover checkpoints
        local checkpoints_str
        checkpoints_str=$(discover_checkpoints "$target_dir")

        if [ -z "$checkpoints_str" ]; then
            log_message "WARNING: No checkpoints found in ${target_dir}"
            continue
        fi

        # Convert to array
        local -a all_checkpoints=()
        while IFS= read -r line; do
            [ -n "$line" ] && all_checkpoints+=("$line")
        done <<< "$checkpoints_str"

        local total_checkpoints=${#all_checkpoints[@]}
        log_message "INFO: Found ${total_checkpoints} checkpoint(s)"

        # Split checkpoints into batches
        split_into_chunks "$ckpt_per_job" "${all_checkpoints[@]}"
        local -a checkpoint_batches=("${CHUNKS[@]}")
        local num_ckpt_batches=${#checkpoint_batches[@]}
        log_message "INFO: Split into ${num_ckpt_batches} checkpoint batch(es)"

        # Parse and optionally split tasks
        parse_tasks_json "$tasks_json"
        local -a task_batches_json=()

        if [ "$split_tasks" = "none" ]; then
            # All tasks in one batch
            task_batches_json+=("$tasks_json")
        elif [ "$split_tasks" = "each" ]; then
            # Each task as separate batch
            for idx in "${!TASK_NAMES_ARR[@]}"; do
                local single_task_json="{\"${TASK_NAMES_ARR[$idx]}\":${TASK_SHOTS_ARR[$idx]}}"
                task_batches_json+=("$single_task_json")
            done
        else
            # Split tasks into groups of N
            local tasks_per_batch="$split_tasks"
            local num_tasks=${#TASK_NAMES_ARR[@]}
            local i=0
            while [ $i -lt $num_tasks ]; do
                local batch_names=()
                local batch_shots=()
                local j=0
                while [ $j -lt $tasks_per_batch ] && [ $((i + j)) -lt $num_tasks ]; do
                    batch_names+=("${TASK_NAMES_ARR[$((i + j))]}")
                    batch_shots+=("${TASK_SHOTS_ARR[$((i + j))]}")
                    ((j++)) || true
                done
                local batch_json=$(tasks_to_json batch_names batch_shots)
                task_batches_json+=("$batch_json")
                ((i += tasks_per_batch)) || true
            done
        fi

        local num_task_batches=${#task_batches_json[@]}
        log_message "INFO: Split into ${num_task_batches} task batch(es)"

        # Generate jobs for each combination of checkpoint batch × task batch
        local batch_idx=0
        for ckpt_batch in "${checkpoint_batches[@]}"; do
            ((++batch_idx))

            local task_idx=0
            for task_batch_json in "${task_batches_json[@]}"; do
                ((++task_idx))

                # Build job name with batch indices
                local job_name="${base_hope_name}"
                if [ $num_ckpt_batches -gt 1 ]; then
                    job_name="${job_name}_ckpt${batch_idx}of${num_ckpt_batches}"
                fi
                if [ $num_task_batches -gt 1 ]; then
                    job_name="${job_name}_task${task_idx}of${num_task_batches}"
                fi

                # Use BASE_OUTPUT_DIR directly - model name will be added by eval_sh.sh
                local full_output_dir="${BASE_OUTPUT_DIR}"

                # Rate limit submissions
                if [ "$MAX_PARALLEL_SUBMISSIONS" -gt 0 ] && [ "${#bg_pids[@]}" -ge "$MAX_PARALLEL_SUBMISSIONS" ]; then
                    log_message "INFO: Reached max parallel submissions. Waiting..."
                    wait -n 2>/dev/null || true
                    # Clean up finished PIDs
                    local new_pids=()
                    for pid in "${bg_pids[@]}"; do
                        if kill -0 "$pid" 2>/dev/null; then
                            new_pids+=("$pid")
                        fi
                    done
                    bg_pids=("${new_pids[@]}")
                fi

                # Submit the job (no command substitution - uses global variable)
                submit_hope_job \
                    "$job_name" \
                    "$full_output_dir" \
                    "$ckpt_batch" \
                    "${task_config[batch_model_type]}" \
                    "${task_config[batch_size]}" \
                    "${task_config[limit]}" \
                    "${task_config[diffusion_eval_mode]}" \
                    "$task_batch_json" \
                    "${task_config[diffusion_type]}" \
                    "${task_config[mc_num]}"

                bg_pids+=($!)
                temp_hope_files+=("$LAST_TEMP_HOPE_FILE")
                ((++total_jobs_submitted))

                log_message "INFO: Submitted ${job_name} (PID: $!, checkpoints: $(echo $ckpt_batch | wc -w))"
            done
        done
    done

    # Wait for all submissions to complete
    log_message "========================================"
    log_message "INFO: Total jobs submitted: ${total_jobs_submitted}"
    log_message "INFO: Waiting for all background processes..."

    if [ ${#bg_pids[@]} -gt 0 ]; then
        wait
        log_message "INFO: All 'hope run' processes completed"
    fi

    # Cleanup temporary files
    log_message "INFO: Cleaning up ${#temp_hope_files[@]} temporary files..."
    for temp_file in "${temp_hope_files[@]}"; do
        rm -f "$temp_file" 2>/dev/null || true
    done

    log_message "--- Eval Job Submission Script V2 Finished ---"
}

# Run main
main "$@"