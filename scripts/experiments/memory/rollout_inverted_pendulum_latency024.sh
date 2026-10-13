#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"

export MUJOCO_GL=glfw
if [[ -z "${DISPLAY:-}" ]]; then
  exec xvfb-run -a bash "$0" "$@"
fi
case "${1:-all}" in
  l0|0) latencies=(0) ;;
  l2|2) latencies=(2) ;;
  l4|4) latencies=(4) ;;
  all) latencies=(0 2 4) ;;
  *) echo "Usage: $0 [l0|l2|l4|all]" >&2; exit 2 ;;
esac
for latency in "${latencies[@]}"; do
  config="configs/experiments/memory/collect/inverted_pendulum_l$latency.yaml"
  "$PYTHON" -m scripts.experiments.memory.collect --config "$config"
done
