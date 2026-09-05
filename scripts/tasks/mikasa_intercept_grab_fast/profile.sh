#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_ARGS=()
if [[ "${1:-}" != --* && "$#" -gt 0 ]]; then
  MODEL_ARGS=(--model-path "$1")
  shift
fi

cd "${REPO_ROOT}"
bash scripts/mikasa/install.sh
exec bash scripts/profile.sh run \
  --model qwenoft \
  --workload mikasa_intercept_grab_fast \
  "${MODEL_ARGS[@]}" \
  --session-python "${REPO_ROOT}/third_party/MIKASA-Robo/.venv/bin/python" \
  "$@"
