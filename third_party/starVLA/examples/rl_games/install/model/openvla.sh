#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_TIER="${STARVLA_INSTALL_TIER:-use}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install peft==0.17.1 "imageio[ffmpeg]==2.37.2" draccus==0.11.5

if [[ "${INSTALL_TIER}" == "dev" ]]; then
  pip_install datasets==4.8.5
fi

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${STARVLA_SKIP_FLASH_ATTN:-false}" == "true" ]]; then
  echo "[install/model/openvla] skipping flash-attn"
else
  PYTHON_BIN="${PYTHON_BIN}" "${INSTALL_DIR}/flash_attn.sh"
fi

echo "[install/model/openvla] tier=${INSTALL_TIER} done"
