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
TRAIN_UPLOAD_REPO_ID="${TRAIN_UPLOAD_REPO_ID:-}"
TRAIN_UPLOAD_REPO_TYPE="${TRAIN_UPLOAD_REPO_TYPE:-}"
TRAIN_UPLOAD_PATH_PREFIX="${TRAIN_UPLOAD_PATH_PREFIX:-}"
EVAL_UPLOAD_REPO_ID="${EVAL_UPLOAD_REPO_ID:-}"
EVAL_UPLOAD_REPO_TYPE="${EVAL_UPLOAD_REPO_TYPE:-}"
EVAL_UPLOAD_PATH_PREFIX="${EVAL_UPLOAD_PATH_PREFIX:-}"

IFS=',' read -r -a ENVS_LIST <<< "${ENVS}"
IFS=',' read -r -a SWEEPS_LIST <<< "${SWEEPS}"
EXTRA_ARGS=("$@")
DEFAULT_UPLOAD_ARGS=()
if [[ -n "${SEEDS}" ]]; then
  EXTRA_ARGS=(--seeds "${SEEDS}" "${EXTRA_ARGS[@]}")
fi
if [[ -n "${TRAIN_UPLOAD_REPO_ID}" ]]; then
  DEFAULT_UPLOAD_ARGS+=(--train-upload-repo-id "${TRAIN_UPLOAD_REPO_ID}")
fi
if [[ -n "${TRAIN_UPLOAD_REPO_TYPE}" ]]; then
  DEFAULT_UPLOAD_ARGS+=(--train-upload-repo-type "${TRAIN_UPLOAD_REPO_TYPE}")
fi
if [[ -n "${TRAIN_UPLOAD_PATH_PREFIX}" ]]; then
  DEFAULT_UPLOAD_ARGS+=(--train-upload-path-prefix "${TRAIN_UPLOAD_PATH_PREFIX}")
fi
if [[ -n "${EVAL_UPLOAD_REPO_ID}" ]]; then
  DEFAULT_UPLOAD_ARGS+=(--eval-upload-repo-id "${EVAL_UPLOAD_REPO_ID}")
fi
if [[ -n "${EVAL_UPLOAD_REPO_TYPE}" ]]; then
  DEFAULT_UPLOAD_ARGS+=(--eval-upload-repo-type "${EVAL_UPLOAD_REPO_TYPE}")
fi
if [[ -n "${EVAL_UPLOAD_PATH_PREFIX}" ]]; then
  DEFAULT_UPLOAD_ARGS+=(--eval-upload-path-prefix "${EVAL_UPLOAD_PATH_PREFIX}")
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
      "${DEFAULT_UPLOAD_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
  done
done
