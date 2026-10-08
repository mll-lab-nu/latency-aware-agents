#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_TIER="${STARVLA_INSTALL_TIER:-use}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install "huggingface-hub>=0.34.0,<1.0" peft safetensors "imageio[ffmpeg]" draccus opencv-python-headless
pip_install diffusers qwen-vl-utils timm tyro

if [[ "${INSTALL_TIER}" == "dev" ]]; then
  pip_install "datasets>=3.0" "pyarrow>=14.0.1" eva-decord av
fi

echo "[install/model/wan_oft] tier=${INSTALL_TIER} done"
