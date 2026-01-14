#!/bin/bash

###########################################################
# Hyperparameter Tuning Script (adapted for QPN/Whisper)
###########################################################

set -euo pipefail

# Default variables
exp_name="qpn_tuning_orion"
output_folder="results_hopt"
hparams="../hparams/whisper_hparam_optim.yaml"
config_file="orion-hpopt.yaml"
hpopt_file="hpopt.yaml"
exp_max_trials=50
store_all=False
compress_exp=False
seed=2028
additional_flags=""
data_folder=""
pretrained_source=""
additional_hparams=""

print_usage() {
    cat <<-USAGE
Usage: $0 [options]

Options:
  --exp_name NAME              Experiment name (default: ${exp_name})
  --output_folder PATH         Output folder for results (default: ${output_folder})
  --hparams PATH               Hparams YAML file (default: ${hparams})
  --config_file PATH           Orion config file (default: ${config_file})
  --hpopt_file PATH            HPopt (orion) config file (default: ${hpopt_file})
  --exp_max_trials INT         Max trials per optimization step (default: ${exp_max_trials})
  --store_all Bool             Store all trial outputs (default: ${store_all})
  --compress_exp Bool          Compress trial outputs (default: ${compress_exp})
  --seed INT                   Random seed (default: ${seed})
  --help                       Show this help message
  Any other flags will be appended to the orion/execution command.
USAGE
}

# Simple argument parsing
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --exp_name) exp_name="$2"; shift 2;;
    --output_folder) output_folder="$2"; shift 2;;
    --hparams) hparams="$2"; shift 2;;
        --data_folder) data_folder="$2"; shift 2;;
        --pretrained_source) pretrained_source="$2"; shift 2;;
        --storage_folder) storage_folder="$2"; shift 2;;
        --feature_size) feature_size="$2" shift 2;;
    --config_file) config_file="$2"; shift 2;;
    --hpopt_file) hpopt_file="$2"; shift 2;;
    --exp_max_trials) exp_max_trials="$2"; shift 2;;
    --store_all) store_all="$2"; shift 2;;
    --compress_exp) compress_exp="$2"; shift 2;;
    --seed) seed="$2"; shift 2;;
    --help) print_usage; exit 0;;
    --*) additional_flags+="$1 $2 "; shift 2;;
    *) POSITIONAL+=("$1"); shift;;
  esac
done

# Build additional_hparams to pass to optimize_hparams.py (e.g. --data_folder, --pretrained_source)
if [ -n "$data_folder" ]; then
    additional_hparams+="--data_folder '$data_folder' "
fi
if [ -n "$pretrained_source" ]; then
    additional_hparams+="--pretrained_source '$pretrained_source' "
fi

if [ -n "$storage_folder" ]; then
    additional_hparams+="--storage_folder '$storage_folder' "
fi

if [ -n "$feature_size" ]; then
    additional_hparams+="--feature_size '$feature_size' "
fi

if [ -z "$output_folder" ] || [ -z "$hparams" ]; then
    echo "ERROR: --output_folder and --hparams are required"
    print_usage
    exit 1
fi

mkdir -p "$output_folder"

echo "Experiment: $exp_name"
echo "Output folder: $output_folder"
echo "Hparams file: $hparams"
echo "Orion config: $config_file"
echo "HPopt file: $hpopt_file"
echo "Extra flags: $additional_flags"

# Helper: extract optimization flags for a given pattern in the YAML
get_flag() {
    local file_path="$1"
    local pattern="$2"
    if [ ! -f "$file_path" ]; then
        echo ""; return 0
    fi
    # grep lines with pattern and strip the pattern prefix
    grep -o "$pattern.*" "$file_path" | sed "s/$pattern//" | tr -d '\n'
}

# Update the yaml using best params (simple key: value replacement)
update_hparams() {
    local best_hparams_file="$1"
    local hparams_yaml_file="$2"
    local output_yaml_file="$3"
    declare -A best
    while IFS=": " read -r key value; do
        best["$key"]="$value"
    done < "$best_hparams_file"

    local content
    content=$(cat "$hparams_yaml_file")
    for k in "${!best[@]}"; do
        # replace the first occurrence of the key: value (very simple)
        content=$(echo "$content" | sed -E "s|^($k:).*|$k: ${best[$k]}|")
    done
    echo "$content" > "$output_yaml_file"
}

# Extract best params block from orion info output
extract_best_params() {
    local info_file="$1"
    local best_line
    best_line=$(grep -n "best trial:" "$info_file" | cut -d: -f1 || true)
    if [ -z "$best_line" ]; then
        echo ""; return
    fi
    # capture params: block until next blank line or 'start time:'
    tail -n +"$best_line" "$info_file" | awk '/params:/{flag=1;next}/start time:/{flag=0}flag' | sed -e 's/^[[:space:]]*//;/^$/d'
}

# Main optimization loop (loop over step ids until no @orion_step<id> found)
step_id=1
hparams_step="$hparams"

while true; do
    pattern="@orion_step${step_id}:"
    opt_flags=$(get_flag "$hparams_step" "$pattern")
    if [ -z "$opt_flags" ]; then
        echo "No optimization flags for step $step_id. Ending loop."
        break
    fi

    output_folder_step="$output_folder/step${step_id}"
    mkdir -p "$output_folder_step"
    exp_name_step="${exp_name}_step${step_id}"

    echo "=============================================="
    echo "Running optimization step $step_id -> $exp_name_step"
    echo "Flags: $opt_flags"
    echo "Output: $output_folder_step"
    echo "=============================================="

    # Build orion command (calls the project's optimize_hparams.py)
    orion_cmd=(orion hunt -n "$exp_name_step" -c "$config_file" --exp-max-trials "$exp_max_trials" python ../optimize_hparams.py "$hparams_step" "$additional_hparams" --hpopt "$hpopt_file" --hpopt_mode orion)

    # Append extracted optimization flags and any additional flags
    eval "orion_cmd+=( $opt_flags $additional_flags )"

    # Save command and run
    printf '%s ' "${orion_cmd[@]}" > "$output_folder_step/orion_hunt_command.txt"
    echo
    echo "Executing: ${orion_cmd[*]}"
    eval "${orion_cmd[*]}"

    # Collect orion info and extract best params
    orion info --name "$exp_name_step" &> "$output_folder_step/orion-info.txt" || true
    best_params_output=$(extract_best_params "$output_folder_step/orion-info.txt")
    best_hparams_file="$output_folder_step/best_hparams.txt"
    echo "$best_params_output" > "$best_hparams_file"

    # Create updated yaml for the next step
    best_yaml_file="$output_folder_step/best_hparams.yaml"
    update_hparams "$best_hparams_file" "$hparams_step" "$best_yaml_file"

    # Move to next step
    hparams_step="$best_yaml_file"
    ((step_id++))
done

echo "Optimization finished. Best yaml available at: $hparams_step"

echo "Done."
