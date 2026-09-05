#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${REPO_ROOT}/outputs/official_gr00t/extreme_parkour_go2_action_yaw_h40_official_flow_aux_dagger/training/extreme_parkour_go2_vla_dagger_action_yaw_h40_official_flow_aux_5k/checkpoint-5000"
SOURCE_ARGS=(--model-path "${MODEL_PATH}")
if [[ "${1:-}" != --* && "$#" -gt 0 ]]; then
  SOURCE_ARGS=(--model-path "$1")
  shift
else
  for ARG in "$@"; do
    case "${ARG}" in
      --hf-artifacts|--model-path|--model-path=*) SOURCE_ARGS=(); break ;;
    esac
  done
fi
SESSION_PYTHON="${PARKOUR_PYTHON:-/pfs/users/lxy/conda/envs/extreme-parkour/bin/python}"

cd "${REPO_ROOT}"
exec bash scripts/profile.sh run \
  --model official_gr00t \
  --workload extreme_parkour_go2 \
  "${SOURCE_ARGS[@]}" \
  --session-python "${SESSION_PYTHON}" \
  "$@"
