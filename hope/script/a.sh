#!/bin/bash
# submit.sh

# --- Global Config ---
HOPE_TEMPLATE="train_debug.hope"   # Your generic hope file
MAIN_SCRIPT="eval.sh"              # The script above
BASE_OUTPUT_DIR="./eval_results"
LOG_FILE="submission.log"

# --- Default Task Params ---
declare -A default_config=(
    [batch_model_type]="discrete_diffusion"
    [batch_size]=16
    [limit]=0
    [diffusion_eval_mode]="mc"
    [tasks_json]='{"hellaswag":3}'
)

# --- Define Your Tasks Here ---
# Each entry is a set of overrides. 
# REQUIRED: 'hope_name' and 'target_parent_dir'
task_overrides=(
    # Task 1: MDLM 1B
    '([hope_name]="eval_mdlm_1B" [target_parent_dir]="/mnt/.../mdlm_1B" [tasks_json]="{\"hellaswag\":10}")'
    
    # Task 2: NIU 1B Causal
    '([hope_name]="eval_niu_1B_causal" [target_parent_dir]="/mnt/.../niu_1B" [batch_model_type]="causal")'
    
    # Task 3: NIU 1B Diffusion MC
    '([hope_name]="eval_niu_1B_mc" [target_parent_dir]="/mnt/.../niu_1B" [diffusion_eval_mode]="mc" [batch_size]=32)'
)

# --- Helper Functions ---
log() { echo "$(date '+%H:%M:%S') | $1" | tee -a "$LOG_FILE"; }

if [ ! -f "$HOPE_TEMPLATE" ]; then
    echo "Error: Template $HOPE_TEMPLATE not found."
    exit 1
fi

echo "--- Starting Submission ---" > "$LOG_FILE"

# --- Main Loop ---
for override_str in "${task_overrides[@]}"; do
    # 1. Reset config to defaults
    unset task_config
    declare -A task_config
    for k in "${!default_config[@]}"; do task_config[$k]="${default_config[$k]}"; done

    # 2. Apply overrides
    eval "declare -A overrides=${override_str}"
    for k in "${!overrides[@]}"; do task_config[$k]="${overrides[$k]}"; done

    # 3. Validation
    name="${task_config[hope_name]}"
    if [ -z "$name" ]; then log "Skipping task: missing hope_name"; continue; fi
    
    target_dir="${task_config[target_parent_dir]}"
    output_dir="${BASE_OUTPUT_DIR}/${name}"
    
    # 4. Construct the Command
    # IMPORTANT: The JSON string needs single quotes around it for Bash.
    cmd="bash ${MAIN_SCRIPT} \"${output_dir}\" \"${target_dir}\" \"${task_config[batch_model_type]}\" \"${task_config[batch_size]}\" \"${task_config[limit]}\" \"${task_config[diffusion_eval_mode]}\" '${task_config[tasks_json]}'"
    
    # 5. Escape for HOPE file
    # We assume the HOPE file expects: worker.script = "YOUR COMMAND"
    # So we must escape any existing double quotes in our command with backslash
    escaped_cmd=${cmd//\"/\\\"}
    
    # 6. Generate specific HOPE file
    generated_hope="${name}.hope"
    log "Generating $generated_hope"
    
    # Use awk to inject the escaped command safely
    awk -v cmd="$escaped_cmd" '
        /^worker.script/ { print "worker.script = \"" cmd "\""; next }
        { print }
    ' "$HOPE_TEMPLATE" > "$generated_hope"

    # 7. Submit
    log "Submitting job: $name"
    # hope run "$generated_hope" 
    
    # Cleanup (Optional: uncomment to remove generated files after submit)
    # rm "$generated_hope"
done

log "Done."