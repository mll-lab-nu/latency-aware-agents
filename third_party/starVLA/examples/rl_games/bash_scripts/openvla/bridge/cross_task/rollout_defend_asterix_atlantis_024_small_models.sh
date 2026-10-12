#!/usr/bin/env bash
# Use the shared fixed-latency collection recipes for these three games.
set -euo pipefail
BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../../.." && pwd)"
cd "$BENCH_ROOT"
PYTHON="${PYTHON:-python}"
for task in defend_the_line asterix atlantis; do
  for latency in 0 2 4; do
    config="configs/experiments/memory/collect/cross_task/${task}_l${latency}.yaml"
    "$PYTHON" -m scripts.experiments.memory.collect --config "$config" --stage collect
    "$PYTHON" -m scripts.experiments.memory.collect --config "$config" --stage publish
  done
done
