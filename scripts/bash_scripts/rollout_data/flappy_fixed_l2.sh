#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"

# The YAML records the teacher, collection budget, probe layout, and publication target.
CONFIG="configs/experiments/memory/collect/flappy_fixed_l2.yaml"
"$PYTHON" -m scripts.experiments.memory.collect --config "$CONFIG" --stage collect
"$PYTHON" -m scripts.experiments.memory.collect --config "$CONFIG" --stage publish
