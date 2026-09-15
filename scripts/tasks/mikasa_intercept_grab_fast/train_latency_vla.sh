#!/usr/bin/env bash
# Usage: DATASET_DIR TEACHER_CHECKPOINT [shared H1 training arguments...]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATASET_DIR="$(realpath "$1")"
TEACHER_CHECKPOINT="$(realpath "$2")"
shift 2
cd "${REPO_ROOT}"
exec "${VLA_PY:-${REPO_ROOT}/.venv/bin/python}" -m scripts.starvla.train_task \
  --task mikasa --model qwenoft \
  --condition latency-aware \
  --dataset-dir "${DATASET_DIR}" --teacher-checkpoint "${TEACHER_CHECKPOINT}" "$@"
