#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/temporal_profiles/flappy_qwenoft_rtx3090}"
PYTHON="${PYTHON:-python3}"
export MPLCONFIGDIR="${OUTPUT_DIR}/.mplconfig"
export XDG_CACHE_HOME="${OUTPUT_DIR}/.cache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "${ROOT}"
mkdir -p "${OUTPUT_DIR}/source"

CASE="experiments/sim2real-calibration/cases/rtx3090/flappy/qwenoft"
PROFILE_REVISION="8531005becf942264a5c8e3965d1a94b657a90df"
DATASET_REVISION="0b3b64a5d5f4a52e5b1decf1a30b01ec8637becc"
if [[ "${SKIP_DOWNLOAD:-0}" != 1 ]]; then
  files=("${CASE}/profile/profile.json" "${CASE}/profile/latency_distribution.json" "${CASE}/profile/latency_burst_model.json")
  for run_id in {0..49}; do
    files+=("${CASE}/real/${run_id}/observation_attempts.jsonl")
  done
  hf download MLL-Lab/LAGEN-profiles "${files[@]}" --repo-type dataset \
    --revision "${PROFILE_REVISION}" --local-dir "${OUTPUT_DIR}/source/profiles"
  hf download MLL-Lab/LAGEN-datasets "${CASE}/experiment.json" --repo-type dataset \
    --revision "${DATASET_REVISION}" --local-dir "${OUTPUT_DIR}/source/experiments"
  cp "${OUTPUT_DIR}/source/experiments/${CASE}/experiment.json" "${OUTPUT_DIR}/source/experiment.json"
fi

"${PYTHON}" -m scripts.experiments.temporal_profiles.compare --output "${OUTPUT_DIR}"
