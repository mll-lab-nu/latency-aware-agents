#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_TIER="${STARVLA_INSTALL_TIER:-use}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install peft "imageio[ffmpeg]" draccus opencv-python-headless

if [[ "${INSTALL_TIER}" == "dev" ]]; then
  pip_install "datasets>=3.0" decord
  pip_install "ray[default]==2.47.0"
fi
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${STARVLA_SKIP_FLASH_ATTN:-false}" == "true" ]]; then
  echo "[install/model/gr00t] skipping flash-attn"
else
  PYTHON_BIN="${PYTHON_BIN}" "${INSTALL_DIR}/flash_attn.sh"
fi

echo "[install/model/gr00t] tier=${INSTALL_TIER} done"
