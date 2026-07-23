#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RAW_ROOT="${REPO_ROOT}/teacher_datasets/memory"
HF_REPO_ID="latency-sensitive-bench/memory-rollouts"

usage() {
  cat >&2 <<'EOF'
Usage: rollout.sh [flappy|demon_attack|deadly_corridor|all]
EOF
}

select_envs() {
  case "$1" in
    flappy|demon_attack|deadly_corridor) ENVS=("$1") ;;
    all) ENVS=(flappy demon_attack deadly_corridor) ;;
    *) usage; exit 2 ;;
  esac
}

manifest_train_artifact_dir() {
  python - "$1" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    print(next(csv.DictReader(handle, delimiter="\t"))["train_artifact_dir"])
PY
}

rollout_env() {
  local env_name="$1"
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
  esac

  raw_name="${env_name}_fix_latency_${latency}_${episodes}ep_7k2steps"
  hf_config_name="${env_name}_fixed_latency_${latency}_${episodes}ep_7k2steps"
  manifest="${REPO_ROOT}/configs/train/memory/rollout_teacher/${env_name}/manifest.tsv"
  checkpoint_root="$(manifest_train_artifact_dir "${manifest}")"

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

select_envs "${1:-all}"
for env_name in "${ENVS[@]}"; do
  rollout_env "${env_name}"
done
