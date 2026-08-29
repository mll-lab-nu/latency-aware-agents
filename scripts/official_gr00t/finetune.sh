#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OFFICIAL_ROOT="${REPO_ROOT}/third_party/Isaac-GR00T"

: "${BASE_MODEL_PATH:?set BASE_MODEL_PATH to an Isaac-GR00T checkpoint}"
: "${DATASET_PATH:?set DATASET_PATH to a converted LeRobot dataset}"
: "${OUTPUT_DIR:?set OUTPUT_DIR for the fine-tuned checkpoint}"
NUM_GPUS="${NUM_GPUS:-1}"

exec uv run --project "${OFFICIAL_ROOT}" python \
    "${OFFICIAL_ROOT}/gr00t/experiment/launch_finetune.py" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${REPO_ROOT}/latency_bench/policy/official_gr00t_modality.py" \
    --num-gpus "${NUM_GPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
