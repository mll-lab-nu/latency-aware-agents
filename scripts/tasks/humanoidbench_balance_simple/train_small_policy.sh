#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ASSET_CACHE_DIR="${BALANCE_ASSET_CACHE_DIR:-${REPO_ROOT}/results/humanoidbench_g1_sonic_assets}"

cd "${REPO_ROOT}"
uv run --project . python -m latency_bench.envs.humanoidbench_sonic_assets \
  --cache-dir "${ASSET_CACHE_DIR}"
uv run --project . python -m training.train \
  --config configs/examples/gymnasium/g1_sonic_balance_simple/fasttd3_train.yaml \
  "$@"
exec uv run --project . python -m latency_bench.run \
  --config configs/examples/gymnasium/g1_sonic_balance_simple/fasttd3_eval.yaml
