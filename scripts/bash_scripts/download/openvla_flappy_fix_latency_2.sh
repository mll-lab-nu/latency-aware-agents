#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://huggingface.co

# Download this archived fixed-latency policy and its original run metadata.
MODEL_REPO=MLL-Lab/LAGEN-models
MODEL_REVISION=0be46c39e73a22721e99558e2be95ad49d5a713d
MODEL_SUBDIR=latency-aware/flappy/vla/starvla-qwenoft-h1/openvla_flappy_fix_latency_2
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/workspace}"

HF_ENDPOINT=https://huggingface.co hf download "${MODEL_REPO}" \
  --revision "${MODEL_REVISION}" \
  --include "${MODEL_SUBDIR}/**" \
  --local-dir "${DOWNLOAD_DIR}/benchmark-models"
