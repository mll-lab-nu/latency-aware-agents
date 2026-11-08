#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export SOURCE_DIR="${SOURCE_DIR:-${ROOT}/outputs/temporal_profiles/flappy_qwenoft_rtx3090_20261107/source}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/temporal_profiles/stronger_flappy_qwenoft_rtx3090}"
export MPLCONFIGDIR="${ROOT}/outputs/temporal_profiles/.mplconfig"
export XDG_CACHE_HOME="${ROOT}/outputs/temporal_profiles/.cache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "${ROOT}"
"${PYTHON:-python3}" -m scripts.experiments.temporal_profiles.study --source-dir "${SOURCE_DIR}" --output "${OUTPUT_DIR}"
