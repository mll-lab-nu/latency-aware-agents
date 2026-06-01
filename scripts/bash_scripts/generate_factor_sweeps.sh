#!/usr/bin/env bash
set -euo pipefail

# usage:
# bash scripts/bash_scripts/generate_factor_sweeps.sh
# SWEEPS=context_window bash scripts/bash_scripts/generate_factor_sweeps.sh --seeds 10,11

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENVS="${ENVS:-flappy,demon_attack,deadly_corridor}"
SWEEPS="${SWEEPS:-context_window,observation_stride}"
SEEDS="${SEEDS:-}"
GENERATED_ROOT="${GENERATED_ROOT:-configs/train/factor_sweeps}"
TRAIN_DIR="${TRAIN_DIR:-results/checkpoints_factor_sweeps}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/factor_sweeps}"
DEVICE="${DEVICE:-gpu}"

IFS=',' read -r -a ENVS_LIST <<< "${ENVS}"
IFS=',' read -r -a SWEEPS_LIST <<< "${SWEEPS}"
EXTRA_ARGS=("$@")
if [[ -n "${SEEDS}" ]]; then
  EXTRA_ARGS=(--seeds "${SEEDS}" "${EXTRA_ARGS[@]}")
fi

for env_name in "${ENVS_LIST[@]}"; do
  if [[ -z "${env_name}" ]]; then
    continue
  fi

  for sweep_name in "${SWEEPS_LIST[@]}"; do
    if [[ -z "${sweep_name}" ]]; then
      continue
    fi

    python scripts/sweeps/generate_flappy_factor_sweeps.py \
      --spec "configs/experiments/${env_name}/factor_sweeps/${sweep_name}/sweep.yaml" \
      --generated-root "${GENERATED_ROOT}/${env_name}/${sweep_name}" \
      --train-dir "${TRAIN_DIR}/${env_name}/${sweep_name}" \
      --output-root "${OUTPUT_ROOT}/${env_name}/${sweep_name}" \
      --device "${DEVICE}" \
      "${EXTRA_ARGS[@]}"
  done
done
