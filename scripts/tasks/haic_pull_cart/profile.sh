#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${REPO_ROOT}/outputs/official_gr00t/haic_pull_cart_vla_action_h40_dagger/training/haic_pull_cart_vla_dagger_action_h40_official_flow_aux_5k/checkpoint-5000"
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
SESSION_PYTHON="${HAIC_PYTHON:-/pfs/users/lxy/conda/envs/haic/bin/python}"

cd "${REPO_ROOT}"
exec bash scripts/profile.sh run \
  --model official_gr00t \
  --workload haic_pull_cart \
  "${SOURCE_ARGS[@]}" \
  --session-python "${SESSION_PYTHON}" \
  "$@"
