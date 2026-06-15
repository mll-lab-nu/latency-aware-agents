#!/usr/bin/env bash
set -euo pipefail

conda activate starvla_rl_games_openvla

export CROSS_DEVICE_RUNTIME_ENV=openvla
export HF_PROFILE_REPO=latency-sensitive-bench/profiles
export HF_PROFILE_REPO_TYPE=dataset

SPEC=configs/experiments/cross_device_protocol.yaml
POLICY=openvla-flappy
PROFILES_DIR=results/cross_device/profiles

python -m scripts.cross_device.protocol measure \
  --spec "$SPEC" \
  --hardware auto \
  --only-policy "$POLICY" \
  --strict-runtime-env \
  --overwrite

python -m scripts.cross_device.protocol sim-same \
  --spec "$SPEC" \
  --hardware auto \
  --only-policy "$POLICY" \
  --strict-runtime-env \
  --profiles-dir "$PROFILES_DIR" \
  --overwirte

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "$GPU_NAME" in
  *H200*|*B200*)
    python -m scripts.cross_device.protocol sim-cross \
      --spec "$SPEC" \
      --replay-hardware auto \
      --only-policy "$POLICY" \
      --strict-runtime-env \
      --profiles-dir "$PROFILES_DIR" \
      --overwrite
    ;;
  *)
    echo "skip sim-cross: replay host is not H200/B200 ($GPU_NAME)"
    ;;
esac
