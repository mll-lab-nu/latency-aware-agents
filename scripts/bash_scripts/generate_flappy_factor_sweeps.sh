#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SPEC="${SPEC:-configs/experiments/flappy/factor_sweeps/sweep.yaml}"
SEEDS="${SEEDS:-}"
GENERATED_ROOT="${GENERATED_ROOT:-configs/train/flappy_factor_sweeps}"
TRAIN_DIR="${TRAIN_DIR:-results/checkpoints_flappy_factor_sweeps}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/flappy_factor_sweeps}"
DEVICE="${DEVICE:-gpu}"

EXTRA_ARGS=("$@")
if [[ -n "${SEEDS}" ]]; then
  EXTRA_ARGS=(--seeds "${SEEDS}" "${EXTRA_ARGS[@]}")
fi

python scripts/sweeps/generate_flappy_factor_sweeps.py \
  --spec "${SPEC}" \
  --generated-root "${GENERATED_ROOT}" \
  --train-dir "${TRAIN_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
