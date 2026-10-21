#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"

case "${1:-all}" in
  asterix|atlantis) tasks=("${1//-/_}"); latencies=(0 2 4) ;;
  all) tasks=(asterix atlantis); latencies=(0 2 4) ;;
  asterix-l[024]|atlantis-l[024])
    tasks=("${1%-l*}"); tasks=("${tasks[0]//-/_}"); latencies=("${1##*-l}") ;;
  *) echo "Usage: $0 [asterix|atlantis|TASK-l0|TASK-l2|TASK-l4|all]" >&2; exit 2 ;;
esac
for task in "${tasks[@]}"; do
  for latency in "${latencies[@]}"; do
    config="configs/experiments/memory/collect/${task}_l${latency}.yaml"
    "$PYTHON" -m scripts.experiments.memory.collect --config "$config"
  done
done
