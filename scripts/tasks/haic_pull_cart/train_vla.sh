#!/usr/bin/env bash
set -euo pipefail

TEACHER_CHECKPOINT="${1:?Usage: $0 TEACHER_CHECKPOINT [train_haic_dagger.py options]}"
shift
BOOTSTRAP=(--run-bootstrap)
if [[ "${1:-}" == "--resume" ]]; then
  BOOTSTRAP=()
  shift
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HAIC_PYTHON="$(command -v python)"

cd "${REPO_ROOT}"
exec "${REPO_ROOT}/third_party/Isaac-GR00T/.venv/bin/python" \
  scripts/official_gr00t/train_haic_dagger.py \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --haic-python "${HAIC_PYTHON}" \
  "${BOOTSTRAP[@]}" "$@"
