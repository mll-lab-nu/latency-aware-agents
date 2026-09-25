#!/usr/bin/env bash
set -euo pipefail

# Generate Demon Attack mixed-latency teacher data for latencies 0, 2, 4, 6.
# For each latency, the script first probes accepted episodes with the clean
# filter, then replays the same accepted seeds twice:
#   1. demon_attack_ghost_trail observations
#   2. raw_rgb observations
#
# On a new machine, edit only this variable if your workspace root differs.
WORKSPACE="${WORKSPACE:-/workspace}"

CHECKPOINT_HF_REPO="placeholder/latency-checkpoints"
CHECKPOINT_HF_INCLUDE="checkpoints_old/demon_attack_sf_same_latency_l6_seed0/**"
CHECKPOINT_LOCAL_DIR="${WORKSPACE}/small_model"
CHECKPOINT_ROOT="${WORKSPACE}/small_model/checkpoints_old/demon_attack_sf_same_latency_l6_seed0"

RUN_ROOT="${WORKSPACE}/outputs/teacher_datasets/demon_attack_mixed_0246_seed_cache"
GHOST_OUTPUT_ROOT="${WORKSPACE}/outputs/teacher_datasets/demon_attack_mixed_0246_ghost30_200ep"
RAW_OUTPUT_ROOT="${WORKSPACE}/outputs/teacher_datasets/demon_attack_mixed_0246_raw_200ep"

GHOST_UPLOAD_HF_REPO="placeholder/demon_attack_mixed_0246_ghost30_200ep"
RAW_UPLOAD_HF_REPO="placeholder/demon_attack_mixed_0246_raw_200ep"

LATENCIES=(0 2 4 6)
EPISODES_PER_LATENCY=50
MAX_STEPS=3600
VAL_FRACTION=0.1
FILTER_PRESET=clean_v1
# Existing demon_attack_200ep accepted-score bins:
#   latency 0: all accepted episodes are >=7310
#   latency 2: all accepted episodes are >=7310
#   latency 4: all accepted episodes are <6460
#   latency 6: all accepted episodes are <6460
# The exact l4/l6 raw thresholds were not saved in the uploaded metadata, so
# keep them explicit and overrideable.
DEMON_EPISODE_THRESHOLD_L0="${DEMON_EPISODE_THRESHOLD_L0:-7310}"
DEMON_EPISODE_THRESHOLD_L2="${DEMON_EPISODE_THRESHOLD_L2:-7310}"
DEMON_EPISODE_THRESHOLD_L4="${DEMON_EPISODE_THRESHOLD_L4:-3500}"
DEMON_EPISODE_THRESHOLD_L6="${DEMON_EPISODE_THRESHOLD_L6:-3500}"
DEVICE=cuda
NUM_ENVS=2048
REPLAY_NUM_ENVS=16
IMAGE_WRITER_WORKERS=32
GHOST_TRAIL_HISTORY_FRAMES=30

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v python >/dev/null 2>&1; then
    echo "error: python not found" >&2
    exit 127
fi
if ! command -v hf >/dev/null 2>&1; then
    echo "error: hf not found" >&2
    exit 127
fi

echo "==> Downloading checkpoint into ${CHECKPOINT_LOCAL_DIR}"
hf download "${CHECKPOINT_HF_REPO}" \
    --repo-type model \
    --include "${CHECKPOINT_HF_INCLUDE}" \
    --local-dir "${CHECKPOINT_LOCAL_DIR}"

if [[ ! -e "${CHECKPOINT_ROOT}" ]]; then
    echo "error: checkpoint root does not exist after download: ${CHECKPOINT_ROOT}" >&2
    exit 1
fi

mkdir -p "${RUN_ROOT}" "${GHOST_OUTPUT_ROOT}" "${RAW_OUTPUT_ROOT}"

write_latency_config() {
    local latency="$1"
    local output_path="$2"
    local fixed_latency_ms
    fixed_latency_ms="$(python - "${latency}" <<'PY'
import sys

latency = int(sys.argv[1])
print(latency * 4 * 1000.0 / 60.0)
PY
)"

    cat >"${output_path}" <<EOF
latency:
  method: fixed
  fixed_latency_ms: ${fixed_latency_ms}
  add_latency_info: false
EOF
}

run_probe() {
    local latency="$1"
    local latency_config="$2"
    local probe_dir="$3"
    local demon_episode_threshold="$4"

    echo "==> Probing latency ${latency} with filter ${FILTER_PRESET}, threshold ${demon_episode_threshold}"
    python scripts/rollout_data/export_sf_teacher_dataset.py \
        --checkpoint-root "${CHECKPOINT_ROOT}" \
        --output-dir "${probe_dir}" \
        --latency-config "${latency_config}" \
        --max-episodes "${EPISODES_PER_LATENCY}" \
        --max-steps-per-episode "${MAX_STEPS}" \
        --val-fraction "${VAL_FRACTION}" \
        --num-envs "${NUM_ENVS}" \
        --replay-num-envs "${REPLAY_NUM_ENVS}" \
        --device "${DEVICE}" \
        --image-writer-workers "${IMAGE_WRITER_WORKERS}" \
        --filter-preset "${FILTER_PRESET}" \
        --demon-episode-threshold "${demon_episode_threshold}" \
        --image-transform raw_rgb \
        --probe-only
}

run_export_from_probe() {
    local latency="$1"
    local latency_config="$2"
    local probe_state="$3"
    local output_dir="$4"
    local image_transform="$5"
    local demon_episode_threshold="$6"

    mkdir -p "${output_dir}"
    cp "${probe_state}" "${output_dir}/probe_state.json"

    echo "==> Replaying latency ${latency} as ${image_transform}"
    if [[ "${image_transform}" == "demon_attack_ghost_trail" ]]; then
        IMAGE_TRANSFORM="${image_transform}" \
        GHOST_TRAIL_HISTORY_FRAMES="${GHOST_TRAIL_HISTORY_FRAMES}" \
        LATENCY_CONFIG="${latency_config}" \
        MAX_EPISODES="${EPISODES_PER_LATENCY}" \
        MAX_STEPS="${MAX_STEPS}" \
        VAL_FRACTION="${VAL_FRACTION}" \
        FILTER_PRESET="${FILTER_PRESET}" \
        DEMON_EPISODE_THRESHOLD="${demon_episode_threshold}" \
        DEVICE="${DEVICE}" \
        NUM_ENVS="${NUM_ENVS}" \
        REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS}" \
        IMAGE_WRITER_WORKERS="${IMAGE_WRITER_WORKERS}" \
        bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
            "${CHECKPOINT_ROOT}" \
            "${output_dir}"
    else
        IMAGE_TRANSFORM="${image_transform}" \
        LATENCY_CONFIG="${latency_config}" \
        MAX_EPISODES="${EPISODES_PER_LATENCY}" \
        MAX_STEPS="${MAX_STEPS}" \
        VAL_FRACTION="${VAL_FRACTION}" \
        FILTER_PRESET="${FILTER_PRESET}" \
        DEMON_EPISODE_THRESHOLD="${demon_episode_threshold}" \
        DEVICE="${DEVICE}" \
        NUM_ENVS="${NUM_ENVS}" \
        REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS}" \
        IMAGE_WRITER_WORKERS="${IMAGE_WRITER_WORKERS}" \
        bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
            "${CHECKPOINT_ROOT}" \
            "${output_dir}"
    fi
}

for latency in "${LATENCIES[@]}"; do
    threshold_var="DEMON_EPISODE_THRESHOLD_L${latency}"
    demon_episode_threshold="${!threshold_var}"
    latency_config="${RUN_ROOT}/latency_${latency}.yaml"
    probe_dir="${RUN_ROOT}/probe_latency_${latency}"
    probe_state="${probe_dir}/probe_state.json"
    ghost_output="${GHOST_OUTPUT_ROOT}/demon_latency_${latency}"
    raw_output="${RAW_OUTPUT_ROOT}/demon_latency_${latency}"

    write_latency_config "${latency}" "${latency_config}"
    run_probe "${latency}" "${latency_config}" "${probe_dir}" "${demon_episode_threshold}"

    if [[ ! -f "${probe_state}" ]]; then
        echo "error: probe state missing for latency ${latency}: ${probe_state}" >&2
        exit 1
    fi

    run_export_from_probe "${latency}" "${latency_config}" "${probe_state}" "${ghost_output}" "demon_attack_ghost_trail" "${demon_episode_threshold}"
    run_export_from_probe "${latency}" "${latency_config}" "${probe_state}" "${raw_output}" "raw_rgb" "${demon_episode_threshold}"

    echo "==> Uploading ghost latency ${latency}"
    hf upload "${GHOST_UPLOAD_HF_REPO}" \
        "${ghost_output}" \
        "demon_latency_${latency}" \
        --repo-type dataset \
        --exclude "sample_factory_init.log" \
        --commit-message "Upload demon attack mixed 0246 ghost30 latency ${latency}"

    echo "==> Uploading raw latency ${latency}"
    hf upload "${RAW_UPLOAD_HF_REPO}" \
        "${raw_output}" \
        "demon_latency_${latency}" \
        --repo-type dataset \
        --exclude "sample_factory_init.log" \
        --commit-message "Upload demon attack mixed 0246 raw latency ${latency}"
done

echo "==> Wrote ghost dataset shards to ${GHOST_OUTPUT_ROOT}"
echo "==> Wrote raw dataset shards to ${RAW_OUTPUT_ROOT}"
