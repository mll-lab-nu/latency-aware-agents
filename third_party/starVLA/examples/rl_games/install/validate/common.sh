#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_TIER="${STARVLA_INSTALL_TIER:-use}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

"$PYTHON_BIN" -m scripts.install.validate starvla --tier "${INSTALL_TIER}"
"${CUDA_HOME}/bin/nvcc" --version | grep -q "release 12.8"
"$PYTHON_BIN" -m latency_bench eval --help >/dev/null
"$PYTHON_BIN" -m compileall "$REPO_ROOT/starVLA/training/rl_games" >/dev/null

echo "[validate/common] tier=${INSTALL_TIER} done"
