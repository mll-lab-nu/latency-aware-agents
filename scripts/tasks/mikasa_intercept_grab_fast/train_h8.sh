#!/usr/bin/env bash
# Usage: train_h8.sh {qwenoft|qwenpi_v3|qwengr00t} [shared training arguments...]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="$1"
VLA_PY_EXEC="$(command -v "${VLA_PY:-python}")"
export CUDA_HOME="$(cd "$(dirname "${VLA_PY_EXEC}")/.." && pwd)"
export PATH="${CUDA_HOME}/bin:${PATH}"
shift
cd "${REPO_ROOT}"
exec "${VLA_PY_EXEC}" -m scripts.starvla.train_task \
  --task mikasa_h8 --model "${MODEL}" "$@" \
  --vla-python "${VLA_PY_EXEC}" --eval-seed 4242424242
