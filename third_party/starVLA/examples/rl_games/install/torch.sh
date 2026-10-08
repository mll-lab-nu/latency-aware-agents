#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_pip.sh
source "${SCRIPT_DIR}/_pip.sh"

echo "[install/torch] Installing the fixed PyTorch 2.11.0 CUDA 12.8 stack"
pip_install \
  torch==2.11.0+cu128 \
  torchvision==0.26.0+cu128 \
  torchaudio==2.11.0+cu128 \
  triton==3.6.0 \
  --index-url "${TORCH_INDEX_URL}"

"$PYTHON_BIN" - <<'PY'
import torch

print(f"[install/torch] torch={torch.__version__} cuda={torch.version.cuda}")
print(f"[install/torch] gpu={torch.cuda.get_device_name(0)}")
PY

echo "[install/torch] done"
