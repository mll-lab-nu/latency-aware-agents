#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SPEC="${SPEC:-}"
SWEEPS="${SWEEPS:-context_window,observation_stride}"
SEEDS="${SEEDS:-}"
GENERATED_ROOT="${GENERATED_ROOT:-configs/train/flappy_factor_sweeps}"
TRAIN_DIR="${TRAIN_DIR:-results/checkpoints_flappy_factor_sweeps}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/flappy_factor_sweeps}"
DEVICE="${DEVICE:-gpu}"

EXTRA_ARGS=("$@")
if [[ -n "${SEEDS}" ]]; then
  EXTRA_ARGS=(--seeds "${SEEDS}" "${EXTRA_ARGS[@]}")
fi

if [[ -n "${SPEC}" ]]; then
  python scripts/sweeps/generate_flappy_factor_sweeps.py \
    --spec "${SPEC}" \
    --generated-root "${GENERATED_ROOT}" \
    --train-dir "${TRAIN_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}"
else
  IFS=',' read -r -a SWEEPS_LIST <<< "${SWEEPS}"
  for sweep_name in "${SWEEPS_LIST[@]}"; do
    if [[ -z "${sweep_name}" ]]; then
      continue
    fi

    python scripts/sweeps/generate_flappy_factor_sweeps.py \
      --spec "configs/experiments/flappy/factor_sweeps/${sweep_name}/sweep.yaml" \
      --generated-root "${GENERATED_ROOT}/${sweep_name}" \
      --train-dir "${TRAIN_DIR}/${sweep_name}" \
      --output-root "${OUTPUT_ROOT}/${sweep_name}" \
      --device "${DEVICE}" \
      "${EXTRA_ARGS[@]}"
  done
fi
