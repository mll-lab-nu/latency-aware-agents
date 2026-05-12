#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENVS="${ENVS:-flappy,demon_attack,deadly_corridor}"
GENERATED_ROOT="${GENERATED_ROOT:-configs/train/factor_sweeps}"
TRAIN_DIR="${TRAIN_DIR:-results/checkpoints_factor_sweeps}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/factor_sweeps}"
DEVICE="${DEVICE:-gpu}"

IFS=',' read -r -a ENVS_LIST <<< "${ENVS}"

for env_name in "${ENVS_LIST[@]}"; do
  if [[ -z "${env_name}" ]]; then
    continue
  fi

  python scripts/sweeps/generate_flappy_factor_sweeps.py \
    --spec "configs/experiments/${env_name}/factor_sweeps/sweep.yaml" \
    --generated-root "${GENERATED_ROOT}/${env_name}" \
    --train-dir "${TRAIN_DIR}/${env_name}" \
    --output-root "${OUTPUT_ROOT}/${env_name}" \
    --device "${DEVICE}" \
    "$@"
done
