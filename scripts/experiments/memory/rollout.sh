#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RAW_ROOT="${REPO_ROOT}/teacher_datasets/memory"
HF_REPO_ID="latency-sensitive-bench/memory-rollouts"

usage() {
  cat >&2 <<'EOF'
Usage: rollout.sh [flappy|demon_attack|deadly_corridor|defend_the_line|all] [latency_raw_frames]
EOF
}

select_envs() {
  case "$1" in
    flappy|demon_attack|deadly_corridor|defend_the_line) ENVS=("$1") ;;
    all) ENVS=(flappy demon_attack deadly_corridor defend_the_line) ;;
    *) usage; exit 2 ;;
  esac
}

manifest_train_artifact_dir() {
  python - "$1" "$2" <<'PY'
import csv
import sys

latency = str(sys.argv[2])
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row["latency_fixed_frames"] == latency:
            print(row["train_artifact_dir"])
            break
    else:
        raise SystemExit(f"No manifest row found for latency_fixed_frames={latency}")
PY
}

rollout_env() {
  local env_name="$1"
  local requested_latency="${2:-}"
  local latency episodes raw_name hf_config_name manifest checkpoint_root

  case "${env_name}" in
    flappy)
      latency=3
      episodes=200
      ;;
    demon_attack)
      latency=6
      episodes=200
      ;;
    deadly_corridor)
      latency=6
      episodes=1000
      ;;
    defend_the_line)
      latency=6
      episodes=1000
      ;;
  esac
  if [[ -n "${requested_latency}" ]]; then
    latency="${requested_latency}"
  fi

  raw_name="${env_name}_fix_latency_${latency}_${episodes}ep_7k2steps"
  hf_config_name="${env_name}_fixed_latency_${latency}_${episodes}ep_7k2steps"
  manifest="${REPO_ROOT}/configs/train/memory/rollout_teacher/${env_name}/manifest.tsv"
  checkpoint_root="$(manifest_train_artifact_dir "${manifest}" "${latency}")"

  MAX_EPISODES="${episodes}" \
  MAX_STEPS=7200 \
  DEVICE=cuda \
  CONTEXT_WINDOW=1 \
  FILTER_PRESET=none \
  HF_REPO_ID="${HF_REPO_ID}" \
  HF_CONFIG_NAME="${hf_config_name}" \
  HF_PRIVATE=0 \
  conda run --no-capture-output -n latency bash \
    "${REPO_ROOT}/scripts/bash_scripts/run_sf_teacher_dataset_export.sh" \
    "${checkpoint_root}" \
    "${RAW_ROOT}/${raw_name}"
}

env_selector="${1:-all}"
latency_selector="${2:-}"
select_envs "${env_selector}"
for env_name in "${ENVS[@]}"; do
  rollout_env "${env_name}" "${latency_selector}"
done
