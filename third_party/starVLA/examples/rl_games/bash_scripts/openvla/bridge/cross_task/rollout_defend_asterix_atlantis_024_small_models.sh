#!/usr/bin/env bash
set -euo pipefail

# Roll out Sample Factory small models for the next cross-task data mix.
#
# Order:
#   1. defend_the_line latencies 0,2,4
#   2. asterix         latencies 0,2,4
#   3. atlantis        latencies 0,2,4
#
# Each latency exports 200 accepted episodes, capped at 7200 raw env frames,
# then publishes named configurations in benchmark-datasets.

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
BENCH_DIR="${BENCH_DIR:-${WORKSPACE_DIR}/latency-sensitive-bench}"
CONDA_ENV="${CONDA_ENV:-latency}"

HF_MODEL_REPO_ID="${HF_MODEL_REPO_ID:-latency-sensitive-bench/paper-experiment-models}"
DEFEND_THE_LINE_HF_REPO="${DEFEND_THE_LINE_HF_REPO:-latency-sensitive-bench/benchmark-datasets}"
ASTERIX_HF_REPO="${ASTERIX_HF_REPO:-latency-sensitive-bench/benchmark-datasets}"
ATLANTIS_HF_REPO="${ATLANTIS_HF_REPO:-latency-sensitive-bench/benchmark-datasets}"
HF_PRIVATE="${HF_PRIVATE:-0}"
HF_MAX_SHARD_SIZE="${HF_MAX_SHARD_SIZE:-500MB}"

LOCAL_MODEL_ROOT="${LOCAL_MODEL_ROOT:-${WORKSPACE_DIR}/hf_small_models/paper-experiment-models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_DIR}/small_model_rollouts/defend_asterix_atlantis_024}"

EPISODES_PER_LATENCY="${EPISODES_PER_LATENCY:-200}"
MAX_STEPS="${MAX_STEPS:-7200}"
VAL_FRACTION="${VAL_FRACTION:-0.01}"
NUM_ENVS="${NUM_ENVS:-12}"
REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS:-16}"
DEVICE="${DEVICE:-cuda}"
EXPORT_IMAGE_SIZE="${EXPORT_IMAGE_SIZE:-224,224}"
IMAGE_WRITER_WORKERS="${IMAGE_WRITER_WORKERS:-16}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-1}"
FILTER_PRESET="${FILTER_PRESET:-none}"
SPLIT_SEED="${SPLIT_SEED:-0}"

# Asterix/Atlantis have five trained seeds in the generated manifest. Use the
# first seed by default; override if you want another checkpoint.
DEFEND_THE_LINE_SEED="${DEFEND_THE_LINE_SEED:-0}"
ASTERIX_SEED="${ASTERIX_SEED:-10}"
ATLANTIS_SEED="${ATLANTIS_SEED:-10}"

LATENCIES=(0 2 4)
ENVS=(defend_the_line asterix atlantis)

usage() {
  cat >&2 <<'EOF'
Usage:
  bash rollout_defend_asterix_atlantis_024_small_models.sh

Expected on the Vast box:
  WORKSPACE_DIR=/workspace
  /workspace/latency-sensitive-bench exists
  conda env "latency" exists
  HF_TOKEN or HUGGINGFACE_HUB_TOKEN is set if the repos are private or uploads require auth

Useful overrides:
  BENCH_DIR=/path/to/latency-sensitive-bench
  CONDA_ENV=latency
  LOCAL_MODEL_ROOT=/mnt/models/paper-experiment-models
  OUTPUT_ROOT=/mnt/data/small_model_rollouts
  NUM_ENVS=24 REPLAY_NUM_ENVS=24 IMAGE_WRITER_WORKERS=24
  FILTER_PRESET=none
  EPISODE_RETURN_GT=1000
  DEFEND_THE_LINE_EPISODE_RETURN_GT=...
  ASTERIX_EPISODE_RETURN_GT=...
  ATLANTIS_EPISODE_RETURN_GT=...
  ASTERIX_SEED=11 ATLANTIS_SEED=14
  HF_PRIVATE=1
  DEFEND_THE_LINE_HF_REPO=latency-sensitive-bench/benchmark-datasets
  ASTERIX_HF_REPO=latency-sensitive-bench/benchmark-datasets
  ATLANTIS_HF_REPO=latency-sensitive-bench/benchmark-datasets
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${BENCH_DIR}" ]]; then
  echo "latency-sensitive-bench not found at BENCH_DIR=${BENCH_DIR}" >&2
  exit 2
fi

cd "${BENCH_DIR}"
mkdir -p "${LOCAL_MODEL_ROOT}" "${OUTPUT_ROOT}"

repo_for_env() {
  case "$1" in
    defend_the_line) echo "${DEFEND_THE_LINE_HF_REPO}" ;;
    asterix) echo "${ASTERIX_HF_REPO}" ;;
    atlantis) echo "${ATLANTIS_HF_REPO}" ;;
    *) echo "Unknown env: $1" >&2; exit 2 ;;
  esac
}

seed_for_env() {
  case "$1" in
    defend_the_line) echo "${DEFEND_THE_LINE_SEED}" ;;
    asterix) echo "${ASTERIX_SEED}" ;;
    atlantis) echo "${ATLANTIS_SEED}" ;;
    *) echo "Unknown env: $1" >&2; exit 2 ;;
  esac
}

manifest_for_env() {
  echo "${BENCH_DIR}/configs/train/memory/small_model/$1/latency_024/manifest.tsv"
}

episode_return_gt_for_env() {
  case "$1" in
    defend_the_line) echo "${DEFEND_THE_LINE_EPISODE_RETURN_GT:-${EPISODE_RETURN_GT:-}}" ;;
    asterix) echo "${ASTERIX_EPISODE_RETURN_GT:-${EPISODE_RETURN_GT:-}}" ;;
    atlantis) echo "${ATLANTIS_EPISODE_RETURN_GT:-${EPISODE_RETURN_GT:-}}" ;;
    *) echo "Unknown env: $1" >&2; exit 2 ;;
  esac
}

manifest_value() {
  local manifest="$1"
  local latency="$2"
  local seed="$3"
  local field="$4"
  python - "$manifest" "$latency" "$seed" "$field" <<'PY'
import csv
import sys

manifest, latency, seed, field = sys.argv[1:5]
with open(manifest, encoding="utf-8", newline="") as handle:
    rows = [
        row for row in csv.DictReader(handle, delimiter="\t")
        if row["latency_fixed_frames"] == latency and row["seed"] == seed
    ]
if not rows:
    raise SystemExit(
        f"No manifest row in {manifest} for latency_fixed_frames={latency}, seed={seed}"
    )
print(rows[0][field])
PY
}

download_model_artifact() {
  local repo_path="$1"
  conda run --no-capture-output -n "${CONDA_ENV}" python - "$HF_MODEL_REPO_ID" "$repo_path" "$LOCAL_MODEL_ROOT" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id, repo_path, local_root = sys.argv[1:4]
Path(local_root).mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    allow_patterns=[f"{repo_path}/**"],
    local_dir=local_root,
)
print(str(Path(local_root) / repo_path))
PY
}

export_one_latency() {
  local env_name="$1"
  local latency="$2"
  local seed repo_id manifest repo_path eval_config checkpoint_root output_dir hf_config_name return_gt hf_data_dir condition task_slug

  seed="$(seed_for_env "${env_name}")"
  repo_id="$(repo_for_env "${env_name}")"
  manifest="$(manifest_for_env "${env_name}")"
  repo_path="$(manifest_value "${manifest}" "${latency}" "${seed}" train_artifact_repo_path)"
  eval_config="$(manifest_value "${manifest}" "${latency}" "${seed}" eval_config)"
  checkpoint_root="${LOCAL_MODEL_ROOT}/${repo_path}"
  output_dir="${OUTPUT_ROOT}/${env_name}_fixed_latency_${latency}_${EPISODES_PER_LATENCY}ep_7k2steps"
  hf_config_name="${env_name}_fixed_latency_${latency}_${EPISODES_PER_LATENCY}ep_7k2steps"
  return_gt="$(episode_return_gt_for_env "${env_name}")"
  task_slug="${env_name//_/-}"
  condition=latency-aware
  if [[ "${latency}" == "0" ]]; then condition=zero-latency; fi
  hf_data_dir="${condition}/${task_slug}/${env_name}_200ep/${hf_config_name}"

  echo
  echo "==> ${env_name} latency=${latency} seed=${seed}"
  echo "    model repo: ${HF_MODEL_REPO_ID}/${repo_path}"
  echo "    dataset:    ${repo_id} config=${hf_config_name}"

  download_model_artifact "${repo_path}"

  cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}"
    python scripts/rollout_data/export_sf_teacher_dataset.py
    --checkpoint-root "${checkpoint_root}"
    --output-dir "${output_dir}"
    --latency-config "${BENCH_DIR}/${eval_config}"
    --max-episodes "${EPISODES_PER_LATENCY}"
    --max-steps-per-episode "${MAX_STEPS}"
    --val-fraction "${VAL_FRACTION}"
    --split-seed "${SPLIT_SEED}"
    --num-envs "${NUM_ENVS}"
    --replay-num-envs "${REPLAY_NUM_ENVS}"
    --device "${DEVICE}"
    --export-image-size "${EXPORT_IMAGE_SIZE}"
    --image-writer-workers "${IMAGE_WRITER_WORKERS}"
    --context-window "${CONTEXT_WINDOW}"
    --filter-preset "${FILTER_PRESET}"
    --push-to-hub
    --hf-repo-id "${repo_id}"
    --hf-config-name "task-transfer--${hf_config_name}"
    --hf-data-dir "${hf_data_dir}"
    --hf-max-shard-size "${HF_MAX_SHARD_SIZE}"
    --hf-token "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
  )

  if [[ "${HF_PRIVATE}" == "1" || "${HF_PRIVATE}" == "true" ]]; then
    cmd+=(--hf-private)
  fi
  if [[ -n "${return_gt}" ]]; then
    cmd+=(--episode-return-gt "${return_gt}")
  fi

  "${cmd[@]}"
}

echo "==> Small-model rollout plan"
echo "    bench:       ${BENCH_DIR}"
echo "    conda env:   ${CONDA_ENV}"
echo "    model repo:  ${HF_MODEL_REPO_ID}"
echo "    output root: ${OUTPUT_ROOT}"
echo "    latencies:   ${LATENCIES[*]}"
echo "    episodes:    ${EPISODES_PER_LATENCY} per latency"
echo "    max steps:   ${MAX_STEPS}"
echo "    filter:      ${FILTER_PRESET}"

for env_name in "${ENVS[@]}"; do
  for latency in "${LATENCIES[@]}"; do
    export_one_latency "${env_name}" "${latency}"
  done
done

echo
echo "==> Done. Pushed:"
echo "    ${DEFEND_THE_LINE_HF_REPO}"
echo "    ${ASTERIX_HF_REPO}"
echo "    ${ATLANTIS_HF_REPO}"
