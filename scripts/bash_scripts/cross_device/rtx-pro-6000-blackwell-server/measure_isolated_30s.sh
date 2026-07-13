#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

# run measure and get raw measurements
bash cross_device.sh measure \
    --config configs/experiments/cross_device_isolated_30s.yaml \
    --hardware rtx-pro-6000-blackwell-server \
    --sessions 0 \
    --resume

# upload raw results
bash cross_device.sh publish raw \
    --config configs/experiments/cross_device_isolated_30s.yaml \
    --hardware rtx-pro-6000-blackwell-server \
    --sessions 0 \
    --resume

# generate profile
TARGET_REF="$(bash cross_device.sh target-ref \
    --config configs/experiments/cross_device_isolated_30s.yaml \
    --hardware rtx-pro-6000-blackwell-server)"

bash cross_device.sh fit \
    --config configs/experiments/cross_device_isolated_30s.yaml \
    --target-ref "$TARGET_REF" \
    --profile-revision fit-0 \
    --sessions 0 \
    --resume

# upload profile
bash cross_device.sh publish profile \
    --config configs/experiments/cross_device_isolated_30s.yaml \
    --profile-ref "$TARGET_REF/fit-0" \
    --resume