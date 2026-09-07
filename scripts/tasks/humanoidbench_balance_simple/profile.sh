#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${REPO_ROOT}/outputs/tasks/humanoidbench_balance_simple/vla_obs198_h40/g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_15k/checkpoint-15000"
SOURCE_ARGS=(--model-path "${MODEL_PATH}")
if [[ "${1:-}" != --* && "$#" -gt 0 ]]; then
  MODEL_PATH="$1"
  shift
  SOURCE_ARGS=(--model-path "${MODEL_PATH}")
else
  for ARG in "$@"; do
    case "${ARG}" in
      --hf-artifacts|--model-path|--model-path=*) SOURCE_ARGS=(); break ;;
    esac
  done
fi
ASSET_CACHE_DIR="${REPO_ROOT}/results/humanoidbench_g1_sonic_assets"

cd "${REPO_ROOT}"
python -m latency_bench.envs.humanoidbench_sonic_assets \
  --cache-dir "${ASSET_CACHE_DIR}"
exec bash scripts/profile.sh run \
  --model official_gr00t \
  --workload g1_sonic_balance_simple \
  "${SOURCE_ARGS[@]}" \
  "$@"
