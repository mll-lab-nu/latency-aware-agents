#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_TIER="${STARVLA_INSTALL_TIER:-use}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install sentencepiece==0.2.1

if [[ "${INSTALL_TIER}" == "dev" ]]; then
  pip_install "datasets>=3.0"
  pip_install jax flax augmax beartype gcsfs openpi-client
  pip_install jaxtyping==0.2.36 tyro==1.0.12 ml-collections==1.0.0 chex==0.1.90 numpydantic==1.8.0
fi
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${STARVLA_SKIP_FLASH_ATTN:-false}" == "true" ]]; then
  echo "[install/model/pi0] skipping flash-attn"
else
  PYTHON_BIN="${PYTHON_BIN}" "${INSTALL_DIR}/flash_attn.sh"
fi

echo "[install/model/pi0] tier=${INSTALL_TIER} done"
