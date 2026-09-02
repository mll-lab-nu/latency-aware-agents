#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
set -a
source "${REPO_ROOT}/.env"
set +a
export PATH="${HAIC_VLA_FFMPEG_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${HAIC_VLA_FFMPEG_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${REPO_ROOT}/third_party/Isaac-GR00T/.venv/bin/python" \
  scripts/official_gr00t/train_haic_dagger.py --run-bootstrap "$@"
