#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${REPO_ROOT}/outputs/tasks/humanoidbench_balance_simple/vla_obs198_h40/checkpoint-15000"
if [[ "${1:-}" != --* && "$#" -gt 0 ]]; then
  MODEL_PATH="$1"
  shift
fi
ASSET_CACHE_DIR="${REPO_ROOT}/results/humanoidbench_g1_sonic_assets"

cd "${REPO_ROOT}"
uv run --project . python -m latency_bench.envs.humanoidbench_sonic_assets \
  --cache-dir "${ASSET_CACHE_DIR}"
exec bash scripts/profile.sh run \
  --model official_gr00t \
  --workload g1_sonic_balance_simple \
  --model-path "${MODEL_PATH}" \
  "$@"
