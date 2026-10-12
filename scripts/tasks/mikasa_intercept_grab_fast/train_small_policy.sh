#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MIKASA_ROOT="${REPO_ROOT}/third_party/MIKASA-Robo"
TASK_PY="${TASK_PY:-python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${MIKASA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${TASK_PY}" \
  "${MIKASA_ROOT}/mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py" \
  --env-id InterceptGrabFast-VLA-v0 \
  --total-timesteps "${MIKASA_TOTAL_TIMESTEPS:-150000000}" \
  "$@"
