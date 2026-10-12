#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m latency_bench collect --config configs/examples/demon_attack/collect_ghost_trail.yaml
if [[ "${REPLAY_RAW_FROM_HF:-0}" == "1" ]]; then
  "$PYTHON" -m scripts.experiments.memory.replay_ghost_trail \
    --config configs/experiments/memory/collect/demon_attack_raw_replay.yaml
fi
