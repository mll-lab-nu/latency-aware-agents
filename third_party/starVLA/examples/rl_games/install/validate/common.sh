#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_TIER="${STARVLA_INSTALL_TIER:-use}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

"$PYTHON_BIN" - <<'PY'
import sys

import torch
import torchaudio
import torchvision
import triton

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torchvision.__version__ == "0.26.0+cu128", torchvision.__version__
assert torchaudio.__version__ == "2.11.0+cu128", torchaudio.__version__
assert triton.__version__ == "3.6.0", triton.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print("ok-fixed-runtime")
PY
"${CUDA_HOME}/bin/nvcc" --version | grep -q "release 12.8"
"$PYTHON_BIN" -c "import omegaconf, torch; print('ok-common-use')"
"$PYTHON_BIN" -c "import starVLA; print('ok-starVLA')"
"$PYTHON_BIN" -m latency_bench.run --help >/dev/null
"$PYTHON_BIN" -m compileall "$REPO_ROOT/starVLA/training/rl_games" >/dev/null

if [[ "${INSTALL_TIER}" == "dev" ]]; then
  "$PYTHON_BIN" -c "import hydra; print('ok-common-dev')"
fi

echo "[validate/common] tier=${INSTALL_TIER} done"
