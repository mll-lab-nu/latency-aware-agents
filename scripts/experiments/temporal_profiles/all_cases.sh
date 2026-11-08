#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export WORKSPACE_DIR="${WORKSPACE_DIR:-${ROOT}}"
export SOURCE_DIR="${SOURCE_DIR:-${WORKSPACE_DIR}/outputs/temporal_profiles/all_cases_source}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/temporal_profiles/all_cases}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${WORKSPACE_DIR}/outputs/temporal_profiles/.mplconfig"
export XDG_CACHE_HOME="${WORKSPACE_DIR}/outputs/temporal_profiles/.cache"
cd "${ROOT}"
if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  "${PYTHON:-python3}" -m scripts.experiments.temporal_profiles.matrix download --source-dir "${SOURCE_DIR}"
fi
"${PYTHON:-python3}" -m scripts.experiments.temporal_profiles.matrix audit --source-dir "${SOURCE_DIR}"
"${PYTHON:-python3}" -m scripts.experiments.temporal_profiles.matrix run --source-dir "${SOURCE_DIR}" --output "${OUTPUT_DIR}" --workers "${WORKERS:-4}"
"${PYTHON:-python3}" -m scripts.experiments.temporal_profiles.matrix_report --output "${OUTPUT_DIR}"
