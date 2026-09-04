#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${REPO_ROOT}/third_party/starVLA/results/Checkpoints/mikasa_grabfast_h1_gb64_mb8_ga8_expandable"
if [[ "${1:-}" != --* && "$#" -gt 0 ]]; then
  MODEL_PATH="$1"
  shift
fi

cd "${REPO_ROOT}"
bash scripts/mikasa/install.sh
exec bash scripts/profile.sh run \
  --model qwenoft \
  --workload mikasa_intercept_grab_fast \
  --model-path "${MODEL_PATH}" \
  --session-python "${REPO_ROOT}/third_party/MIKASA-Robo/.venv/bin/python" \
  "$@"
