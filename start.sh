#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="${CONFIG:-configs/experiments/cross_device.yaml}"
HARDWARE="${HARDWARE:-auto}"
PROFILE_REVISION="${PROFILE_REVISION:-fit-0}"
METHOD="${METHOD:-temporal}"
read -r -a SESSIONS <<< "${SESSIONS:-0}"
read -r -a FIT_SESSIONS <<< "${FIT_SESSIONS:-${SESSIONS[*]}}"
read -r -a LATENCY_SEEDS <<< "${LATENCY_SEEDS:-0 1 2}"

TARGET_REF="$(bash cross_device.sh target-ref --config "$CONFIG" --hardware "$HARDWARE")"
PROFILE_REF="${TARGET_REF}/${PROFILE_REVISION}"

bash cross_device.sh measure --config "$CONFIG" --hardware "$HARDWARE" \
  --sessions "${SESSIONS[@]}" --resume
bash cross_device.sh publish raw --config "$CONFIG" --hardware "$HARDWARE" \
  --sessions "${SESSIONS[@]}" --resume
bash cross_device.sh fit --config "$CONFIG" --target-ref "$TARGET_REF" \
  --profile-revision "$PROFILE_REVISION" --sessions "${FIT_SESSIONS[@]}" --resume
bash cross_device.sh publish profile --config "$CONFIG" \
  --profile-ref "$PROFILE_REF" --resume
bash cross_device.sh sim --config "$CONFIG" --profile-ref "$PROFILE_REF" \
  --method "$METHOD" --latency-seeds "${LATENCY_SEEDS[@]}" --resume
bash cross_device.sh publish sim --config "$CONFIG" --profile-ref "$PROFILE_REF" \
  --method "$METHOD" --latency-seeds "${LATENCY_SEEDS[@]}" --resume
