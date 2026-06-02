#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEEDS="${SEEDS:-911,922,933}"
GENERATED_ROOT="${GENERATED_ROOT:-configs/train/jitter_robustness/flappy}"
TRAIN_DIR="${TRAIN_DIR:-results/checkpoints_jitter_robustness/flappy}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/jitter_robustness/flappy}"
DEVICE="${DEVICE:-gpu}"

python scripts/sweeps/generate_flappy_factor_sweeps.py \
  --spec configs/experiments/flappy/jitter_robustness/sweep.yaml \
  --generated-root "${GENERATED_ROOT}" \
  --train-dir "${TRAIN_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "${DEVICE}" \
  --seeds "${SEEDS}" \
  "$@"
