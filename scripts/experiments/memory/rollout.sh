#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"

if [[ $# -gt 1 ]]; then
  echo "Select the latency in configs/experiments/memory/collect/*.yaml" >&2
  exit 2
fi
case "${1:-all}" in
  flappy|demon_attack|deadly_corridor) tasks=("$1") ;;
  all) tasks=(flappy demon_attack deadly_corridor) ;;
  *) echo "Usage: $0 [flappy|demon_attack|deadly_corridor|all]" >&2; exit 2 ;;
esac
for task in "${tasks[@]}"; do
  config="configs/experiments/memory/collect/$task.yaml"
  "$PYTHON" -m scripts.experiments.memory.collect --config "$config"
done
