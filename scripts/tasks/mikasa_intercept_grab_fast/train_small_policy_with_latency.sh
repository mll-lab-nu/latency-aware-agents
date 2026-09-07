#!/usr/bin/env bash
# Train or evaluate the native small policy with simulated latency.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEACHER_CHECKPOINT="${1:?usage: $0 TEACHER_CHECKPOINT LATENCY_CONFIG OUTPUT_DIR [native args ...]}"
LATENCY_CONFIG="${2:?latency config required}"
OUTPUT_DIR="${3:?output directory required}"
shift 3

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/MIKASA-Robo${PYTHONPATH:+:${PYTHONPATH}}"
exec third_party/MIKASA-Robo/.venv/bin/python \
  third_party/MIKASA-Robo/mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py \
  --env-id InterceptGrabFast-VLA-v0 \
  --checkpoint "${TEACHER_CHECKPOINT}" \
  --latency-config "${LATENCY_CONFIG}" \
  --latency-output-dir "${OUTPUT_DIR}" "$@"
