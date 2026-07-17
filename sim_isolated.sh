#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

readonly CONFIG=configs/experiments/cross_device_isolated_30s.yaml
readonly ARTIFACT_ROOT=results/cross_device
readonly WORKLOAD_ID=openvla-flappy-isolated-30s
readonly PROFILE_REVISION=fit-0
readonly -a GPU_SLOTS=("0,1" "2,3" "4,5" "6,7")
readonly -a METHODS=(temporal iid fixed normal)
readonly -a DEVICES=(l40s rtx3090 rtx4090 rtx5090 a100-sxm4-80gb)
declare -Ar INSTANCES=(
    [l40s]=instance_edd21c9a
    [rtx3090]=instance_000df10e
    [rtx4090]=instance_e4777ea2
    [rtx5090]=instance_02874a5a
    [a100-sxm4-80gb]=instance_1d6030f8
)

for device in "${DEVICES[@]}"; do
    target_ref="${device}/${INSTANCES[$device]}/${WORKLOAD_ID}"
    profile_ref="${target_ref}/${PROFILE_REVISION}"

    bash cross_device.sh download raw \
        --config "$CONFIG" \
        --target-ref "$target_ref" \
        --sessions 0 \
        --overwrite
    rm -rf -- "${ARTIFACT_ROOT}/profiles/${profile_ref}"
    bash cross_device.sh fit \
        --config "$CONFIG" \
        --target-ref "$target_ref" \
        --profile-revision "$PROFILE_REVISION" \
        --sessions 0
    bash cross_device.sh publish profile \
        --config "$CONFIG" \
        --profile-ref "$profile_ref" \
        --overwrite
done

printf '%s\n' 'Preparing evaluation weights...'
RUN_INSIDE_TMUX=1 bash scripts/bash_scripts/download/openvla_flappy_fix_latency_2.sh
RUN_INSIDE_TMUX=1 bash scripts/bash_scripts/download/qwen3_vl_4b_instruct.sh
bash scripts/bash_scripts/download/link_eval_paths.sh

declare -a TASK_PROFILE_REFS=()
declare -a TASK_METHODS=()
declare -a TASK_LABELS=()
for device in "${DEVICES[@]}"; do
    profile_ref="${device}/${INSTANCES[$device]}/${WORKLOAD_ID}/${PROFILE_REVISION}"
    for method in "${METHODS[@]}"; do
        TASK_PROFILE_REFS+=("$profile_ref")
        TASK_METHODS+=("$method")
        TASK_LABELS+=("${profile_ref}/${method}/0")
    done
done

declare -A PID_SLOTS=()
declare -A PID_TASKS=()
declare -a FAILURES=()
next_task=0
running_tasks=0

launch_task() {
    local slot="$1"
    local task_index="$2"
    local profile_ref="${TASK_PROFILE_REFS[$task_index]}"
    local method="${TASK_METHODS[$task_index]}"
    local label="${TASK_LABELS[$task_index]}"

    (
        printf '[slot %s] start %s\n' "$slot" "$label"
        rm -rf -- "${ARTIFACT_ROOT}/sim/${profile_ref}/${method}/0"
        CUDA_VISIBLE_DEVICES="$slot" bash cross_device.sh sim \
            --config "$CONFIG" \
            --profile-ref "$profile_ref" \
            --method "$method" \
            --latency-seeds 0 || exit $?
        CUDA_VISIBLE_DEVICES="$slot" bash cross_device.sh publish sim \
            --config "$CONFIG" \
            --profile-ref "$profile_ref" \
            --method "$method" \
            --latency-seeds 0 \
            --overwrite || exit $?
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

(
    cd paper/latency-sensitive-vla/exps/sim2real
    python build_data.py
    python plot.py
)
