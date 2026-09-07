#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ASSET_CACHE_DIR="${BALANCE_ASSET_CACHE_DIR:-${REPO_ROOT}/results/humanoidbench_g1_sonic_assets}"
TRAIN_CONFIG="${BALANCE_TRAIN_CONFIG:-configs/examples/gymnasium/g1_sonic_balance_simple/fasttd3_train.yaml}"
EVAL_CONFIG="${BALANCE_EVAL_CONFIG:-configs/examples/gymnasium/g1_sonic_balance_simple/fasttd3_eval.yaml}"
TRAIN_DIR_ARGS=()
if [[ -n "${BALANCE_TRAIN_DIR:-}" ]]; then
  TRAIN_DIR_ARGS+=(--train-dir "${BALANCE_TRAIN_DIR}")
fi

cd "${REPO_ROOT}"
uv run --project . python -m latency_bench.envs.humanoidbench_sonic_assets \
  --cache-dir "${ASSET_CACHE_DIR}"
uv run --project . python -m training.train \
  --config "${TRAIN_CONFIG}" \
  "${TRAIN_DIR_ARGS[@]}" \
  "$@"
EVAL_ARGS=()
if [[ -n "${BALANCE_TRAIN_RUN_NAME:-}" ]]; then
  EVAL_ARGS+=(--run-name "${BALANCE_TRAIN_RUN_NAME}" --checkpoint-kind latest)
fi
exec uv run --project . python -m latency_bench.run \
  --config "${EVAL_CONFIG}" \
  "${EVAL_ARGS[@]}"
