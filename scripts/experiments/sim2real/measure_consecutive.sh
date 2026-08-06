#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

HARDWARE="$1"
CONFIG="configs/experiments/cross_device.yaml"

bash "${SCRIPT_DIR}/cross_device.sh" measure \
    --config "${CONFIG}" \
    --hardware "${HARDWARE}" \
    --sessions 0 1 2 3 4 \
    --resume

bash "${SCRIPT_DIR}/cross_device.sh" publish raw \
    --config "${CONFIG}" \
    --hardware "${HARDWARE}" \
    --sessions 0 1 2 3 4 \
    --resume

TARGET_REF="$(bash "${SCRIPT_DIR}/cross_device.sh" target-ref \
    --config "${CONFIG}" \
    --hardware "${HARDWARE}")"

bash "${SCRIPT_DIR}/cross_device.sh" fit \
    --config "${CONFIG}" \
    --target-ref "${TARGET_REF}" \
    --profile-revision fit-0-4 \
    --sessions 0 1 2 3 4 \
    --resume

bash "${SCRIPT_DIR}/cross_device.sh" publish profile \
    --config "${CONFIG}" \
    --profile-ref "${TARGET_REF}/fit-0-4" \
    --resume
