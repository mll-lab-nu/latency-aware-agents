#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${REPO_ROOT}/outputs/official_gr00t/haic_pull_cart_vla_action_h40_dagger/training/haic_pull_cart_vla_dagger_action_h40_official_sampler_5k/checkpoint-5000"
if [[ "${1:-}" != --* && "$#" -gt 0 ]]; then
  MODEL_PATH="$1"
  shift
fi
SESSION_PYTHON="$(command -v python)"

cd "${REPO_ROOT}"
exec bash scripts/profile.sh run \
  --model official_gr00t \
  --workload haic_pull_cart \
  --model-path "${MODEL_PATH}" \
  --session-python "${SESSION_PYTHON}" \
  "$@"
