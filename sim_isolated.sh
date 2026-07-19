#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

readonly ISOLATED_CONFIG=configs/experiments/cross_device_isolated_30s.yaml
readonly CONSECUTIVE_CONFIG=configs/experiments/cross_device.yaml
readonly ARTIFACT_ROOT=results/cross_device
readonly ISOLATED_WORKLOAD_ID=openvla-flappy-isolated-30s
readonly ISOLATED_PROFILE_REVISION=fit-0
readonly CONSECUTIVE_TARGET_REF=rtx4090/instance_9fb8d975/openvla-flappy
readonly CONSECUTIVE_PROFILE_REVISION=fit-0-4
readonly CONSECUTIVE_PROFILE_REF="${CONSECUTIVE_TARGET_REF}/${CONSECUTIVE_PROFILE_REVISION}"
readonly -a GPU_SLOTS=("0,1" "2,3" "4,5" "6,7")
readonly -a METHODS=(temporal iid fixed normal)
readonly -a CONSECUTIVE_SEEDS=(0 1 2 3 4)
readonly -a ISOLATED_DEVICES=(l40s rtx3090 rtx4090 rtx5090 a100-sxm4-80gb)
declare -Ar ISOLATED_INSTANCES=(
    [l40s]=instance_edd21c9a
    [rtx3090]=instance_000df10e
    [rtx4090]=instance_e4777ea2
    [rtx5090]=instance_02874a5a
    [a100-sxm4-80gb]=instance_1d6030f8
)
declare -a REBUILT_PROFILE_CONFIGS=()
declare -a REBUILT_PROFILE_REFS=()

rebuild_profile() {
    local config="$1"
    local target_ref="$2"
    local profile_revision="$3"
    shift 3
    local -a sessions=("$@")
    local profile_ref="${target_ref}/${profile_revision}"

    bash cross_device.sh download raw \
        --config "$config" \
        --target-ref "$target_ref" \
        --sessions "${sessions[@]}" \
        --overwrite
    rm -rf -- "${ARTIFACT_ROOT}/profiles/${profile_ref}"
    bash cross_device.sh fit \
        --config "$config" \
        --target-ref "$target_ref" \
        --profile-revision "$profile_revision" \
        --sessions "${sessions[@]}"
    REBUILT_PROFILE_CONFIGS+=("$config")
    REBUILT_PROFILE_REFS+=("$profile_ref")
}

printf '%s\n' 'Rebuilding RTX 4090 consecutive profile from HF raw sessions 0-4...'
rebuild_profile \
    "$CONSECUTIVE_CONFIG" \
    "$CONSECUTIVE_TARGET_REF" \
    "$CONSECUTIVE_PROFILE_REVISION" \
    "${CONSECUTIVE_SEEDS[@]}"

printf '%s\n' 'Rebuilding isolated profiles from HF raw session 0...'
for device in "${ISOLATED_DEVICES[@]}"; do
    target_ref="${device}/${ISOLATED_INSTANCES[$device]}/${ISOLATED_WORKLOAD_ID}"
    rebuild_profile \
        "$ISOLATED_CONFIG" \
        "$target_ref" \
        "$ISOLATED_PROFILE_REVISION" \
        0
done

printf '%s\n' 'Preparing evaluation weights...'
RUN_INSIDE_TMUX=1 bash scripts/bash_scripts/download/openvla_flappy_fix_latency_2.sh
RUN_INSIDE_TMUX=1 bash scripts/bash_scripts/download/qwen3_vl_4b_instruct.sh
bash scripts/bash_scripts/download/link_eval_paths.sh

declare -a TASK_CONFIGS=()
declare -a TASK_PROFILE_REFS=()
declare -a TASK_METHODS=()
declare -a TASK_SEEDS=()
declare -a TASK_LABELS=()

append_simulation_tasks() {
    local config="$1"
    local profile_ref="$2"
    shift 2
    local method
    local latency_seed

    for method in "${METHODS[@]}"; do
        for latency_seed in "$@"; do
            TASK_CONFIGS+=("$config")
            TASK_PROFILE_REFS+=("$profile_ref")
            TASK_METHODS+=("$method")
            TASK_SEEDS+=("$latency_seed")
            TASK_LABELS+=("${profile_ref}/${method}/${latency_seed}")
        done
    done
}

append_simulation_tasks \
    "$CONSECUTIVE_CONFIG" \
    "$CONSECUTIVE_PROFILE_REF" \
    "${CONSECUTIVE_SEEDS[@]}"
for device in "${ISOLATED_DEVICES[@]}"; do
    profile_ref="${device}/${ISOLATED_INSTANCES[$device]}/${ISOLATED_WORKLOAD_ID}/${ISOLATED_PROFILE_REVISION}"
    append_simulation_tasks "$ISOLATED_CONFIG" "$profile_ref" 0
done

declare -A PID_SLOTS=()
declare -A PID_TASKS=()
declare -a FAILURES=()
next_task=0
running_tasks=0

launch_task() {
    local slot="$1"
    local task_index="$2"
    local config="${TASK_CONFIGS[$task_index]}"
    local profile_ref="${TASK_PROFILE_REFS[$task_index]}"
    local method="${TASK_METHODS[$task_index]}"
    local latency_seed="${TASK_SEEDS[$task_index]}"
    local label="${TASK_LABELS[$task_index]}"

    (
        printf '[slot %s] start %s\n' "$slot" "$label"
        rm -rf -- "${ARTIFACT_ROOT}/sim/${profile_ref}/${method}/${latency_seed}"
        CUDA_VISIBLE_DEVICES="$slot" bash cross_device.sh sim \
            --config "$config" \
            --profile-ref "$profile_ref" \
            --method "$method" \
            --latency-seeds "$latency_seed" \
            --local-profile || exit $?
        printf '[slot %s] done %s\n' "$slot" "$label"
    ) &

    local pid=$!
    PID_SLOTS["$pid"]="$slot"
    PID_TASKS["$pid"]="$task_index"
    running_tasks=$((running_tasks + 1))
}

for slot in "${GPU_SLOTS[@]}"; do
    launch_task "$slot" "$next_task"
    next_task=$((next_task + 1))
done

while ((running_tasks > 0)); do
    finished_pid=""
    if wait -n -p finished_pid; then
        task_status=0
    else
        task_status=$?
    fi

    slot="${PID_SLOTS[$finished_pid]}"
    task_index="${PID_TASKS[$finished_pid]}"
    unset 'PID_SLOTS[$finished_pid]'
    unset 'PID_TASKS[$finished_pid]'
    running_tasks=$((running_tasks - 1))

    if ((task_status != 0)); then
        FAILURES+=("${TASK_LABELS[$task_index]} (slot ${slot}, exit ${task_status})")
    fi

    if ((next_task < ${#TASK_LABELS[@]})); then
        launch_task "$slot" "$next_task"
        next_task=$((next_task + 1))
    fi
done

if ((${#FAILURES[@]} > 0)); then
    printf '%s\n' 'Simulation tasks failed:' >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

printf '%s\n' 'Publishing rebuilt profiles...'
for profile_index in "${!REBUILT_PROFILE_REFS[@]}"; do
    bash cross_device.sh publish profile \
        --config "${REBUILT_PROFILE_CONFIGS[$profile_index]}" \
        --profile-ref "${REBUILT_PROFILE_REFS[$profile_index]}" \
        --overwrite
done

printf '%s\n' 'Publishing completed simulation matrix...'
for task_index in "${!TASK_LABELS[@]}"; do
    bash cross_device.sh publish sim \
        --config "${TASK_CONFIGS[$task_index]}" \
        --profile-ref "${TASK_PROFILE_REFS[$task_index]}" \
        --method "${TASK_METHODS[$task_index]}" \
        --latency-seeds "${TASK_SEEDS[$task_index]}" \
        --overwrite
done

(
    cd paper/latency-sensitive-vla/exps/sim2real
    python build_data.py
    python plot.py
)
