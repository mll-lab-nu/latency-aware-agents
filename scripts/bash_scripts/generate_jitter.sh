#!/usr/bin/env bash
set -euo pipefail

# usage:
# bash scripts/bash_scripts/generate_jitter.sh --seeds 0,42,114514

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENVS="${ENVS:-flappy,demon_attack,deadly_corridor}"
SEEDS="${SEEDS:-}"
GENERATED_ROOT="${GENERATED_ROOT:-configs/train/jitter_sweeps}"
TRAIN_DIR="${TRAIN_DIR:-results/checkpoints_jitter_sweeps}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/jitter_sweeps}"
DEVICE="${DEVICE:-gpu}"

IFS=',' read -r -a ENVS_LIST <<< "${ENVS}"
EXTRA_ARGS=("$@")
if [[ -n "${SEEDS}" ]]; then
  EXTRA_ARGS=(--seeds "${SEEDS}" "${EXTRA_ARGS[@]}")
fi

for env_name in "${ENVS_LIST[@]}"; do
  if [[ -z "${env_name}" ]]; then
    continue
  fi

  python scripts/sweeps/generate_flappy_factor_sweeps.py \
    --spec "configs/experiments/${env_name}/jitter/sweep.yaml" \
    --generated-root "${GENERATED_ROOT}/${env_name}" \
    --train-dir "${TRAIN_DIR}/${env_name}" \
    --output-root "${OUTPUT_ROOT}/${env_name}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}"
done
