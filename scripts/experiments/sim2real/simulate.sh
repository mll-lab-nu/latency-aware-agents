#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

readonly -a GPU_SLOTS=("0,1" "2,3" "4,5" "6,7")
readonly -a METHODS=(temporal iid fixed normal)
readonly -a DEVICES=(l40s rtx3090 rtx4090 rtx5090 a100-sxm4-80gb)
declare -Ar ISOLATED_INSTANCES=(
    [l40s]=instance_edd21c9a
    [rtx3090]=instance_000df10e
    [rtx4090]=instance_e4777ea2
    [rtx5090]=instance_02874a5a
    [a100-sxm4-80gb]=instance_1d6030f8
)
declare -Ar CONSECUTIVE_INSTANCES=(
    [l40s]=instance_cfa273e0
    [rtx3090]=instance_000df10e
    [rtx4090]=instance_9fb8d975
    [rtx5090]=instance_02874a5a
    [a100-sxm4-80gb]=instance_1d6030f8
)

declare -a TASK_CONFIGS=()
declare -a TASK_PROFILE_REFS=()
declare -a TASK_METHODS=()
declare -a TASK_SEEDS=()
declare -a TASK_LABELS=()

append_lifecycle_tasks() {
    local config="$1"
    local workload_id="$2"
    local profile_revision="$3"
    local -n instances="$4"
    shift 4
    local -a latency_seeds=("$@")
    local device
    local profile_ref
    local method
    local latency_seed

    for device in "${DEVICES[@]}"; do
        profile_ref="${device}/${instances[$device]}/${workload_id}/${profile_revision}"
        for method in "${METHODS[@]}"; do
            for latency_seed in "${latency_seeds[@]}"; do
                TASK_CONFIGS+=("$config")
                TASK_PROFILE_REFS+=("$profile_ref")
                TASK_METHODS+=("$method")
                TASK_SEEDS+=("$latency_seed")
                TASK_LABELS+=("${profile_ref}/${method}/${latency_seed}")
            done
        done
    done
}

append_lifecycle_tasks \
    configs/experiments/cross_device.yaml \
    openvla-flappy \
    fit-0-4 \
    CONSECUTIVE_INSTANCES \
    0 1 2 3 4
append_lifecycle_tasks \
    configs/experiments/cross_device_isolated_30s.yaml \
    openvla-flappy-isolated-30s \
    fit-0 \
    ISOLATED_INSTANCES \
    0

prepare_weights() {
    printf '%s\n' 'Preparing evaluation weights...'
    RUN_INSIDE_TMUX=1 bash scripts/bash_scripts/download/openvla_flappy_fix_latency_2.sh
    RUN_INSIDE_TMUX=1 bash scripts/bash_scripts/download/qwen3_vl_4b_instruct.sh
    bash scripts/bash_scripts/download/link_eval_paths.sh
}

declare -A PID_SLOTS=()
declare -A PID_TASKS=()
declare -a FAILURES=()

next_task=0
running_tasks=0

prepare_weights

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
        CUDA_VISIBLE_DEVICES="$slot" bash "${SCRIPT_DIR}/cross_device.sh" sim \
            --config "$config" \
            --profile-ref "$profile_ref" \
            --method "$method" \
            --latency-seeds "$latency_seed" \
            --resume || exit $?
        CUDA_VISIBLE_DEVICES="$slot" bash "${SCRIPT_DIR}/cross_device.sh" publish sim \
            --config "$config" \
            --profile-ref "$profile_ref" \
            --method "$method" \
            --latency-seeds "$latency_seed" \
            --resume || exit $?
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
