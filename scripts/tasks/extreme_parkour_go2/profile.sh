#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${REPO_ROOT}/outputs/official_gr00t/extreme_parkour_go2_action_yaw_h40_official_sampler_dagger/training/extreme_parkour_go2_vla_dagger_action_yaw_h40_official_sampler_5k/checkpoint-5000"
if [[ "${1:-}" != --* && "$#" -gt 0 ]]; then
  MODEL_PATH="$1"
  shift
fi
SESSION_PYTHON="$(command -v python)"

cd "${REPO_ROOT}"
exec bash scripts/profile.sh run \
  --model official_gr00t \
  --workload extreme_parkour_go2 \
  --model-path "${MODEL_PATH}" \
  --session-python "${SESSION_PYTHON}" \
  "$@"
