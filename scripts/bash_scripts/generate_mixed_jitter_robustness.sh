#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENVS="${ENVS:-flappy,demon_attack,deadly_corridor}"
SEEDS="${SEEDS:-}"
GENERATED_ROOT="${GENERATED_ROOT:-configs/train/jitter_robustness}"
TRAIN_DIR="${TRAIN_DIR:-results/checkpoints_jitter_robustness}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/jitter_robustness}"
DEVICE="${DEVICE:-gpu}"

IFS=',' read -r -a ENVS_LIST <<< "${ENVS}"
EXTRA_ARGS=("$@")

for env_name in "${ENVS_LIST[@]}"; do
  if [[ -z "${env_name}" ]]; then
    continue
  fi

  env_seeds="${SEEDS}"
  if [[ -z "${env_seeds}" ]]; then
    if [[ "${env_name}" == "flappy" ]]; then
      env_seeds="911,922,933"
    else
      env_seeds="0,42,114514"
    fi
  fi

  python scripts/sweeps/generate_flappy_factor_sweeps.py \
    --spec "configs/experiments/${env_name}/jitter_robustness/mixed_sweeps.yaml" \
    --generated-root "${GENERATED_ROOT}/${env_name}/mixed" \
    --train-dir "${TRAIN_DIR}/${env_name}/mixed" \
    --output-root "${OUTPUT_ROOT}/${env_name}/mixed" \
    --device "${DEVICE}" \
    --seeds "${env_seeds}" \
    "${EXTRA_ARGS[@]}"
done
