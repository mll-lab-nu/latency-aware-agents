#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."

# run measure and get raw measurements
bash cross_device.sh measure \
    --config configs/experiments/cross_device_isolated_30s.yaml \
    --hardware a100-sxm4-40gb \
    --sessions 0 1 2 3 4 \
    --resume

# upload raw results
bash cross_device.sh publish raw \
    --config configs/experiments/cross_device_isolated_30s.yaml \
    --hardware a100-sxm4-40gb \
    --sessions 0 1 2 3 4 \
    --resume

# generate profile
TARGET_REF="$(bash cross_device.sh target-ref \
    --config configs/experiments/cross_device_isolated_30s.yaml \
    --hardware a100-sxm4-40gb)"

bash cross_device.sh fit \
    --config configs/experiments/cross_device_isolated_30s.yaml \
    --target-ref "$TARGET_REF" \
    --profile-revision fit-0-4 \
    --sessions 0 1 2 3 4 \
    --resume

# upload profile
bash cross_device.sh publish profile \
    --config configs/experiments/cross_device_isolated_30s.yaml \
    --profile-ref "$TARGET_REF/fit-0-4" \
    --resume