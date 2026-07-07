#!/usr/bin/env bash
# Export a Flappy teacher rollout dataset for KV-memory training.
# Step 1: download the factor-sweep teacher checkpoint from the Hugging Face hub.
# Step 2: replay it and export canonical rollout rows at the observation cadence.
set -euo pipefail

# ---- run variables ----
MAX_EPISODES=200
MAX_STEPS=7200
NUM_ENVS="${NUM_ENVS:-4096}"
REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS:-32}"
DEVICE=gpu
IMAGE_WRITER_WORKERS=32
# HF dataset target: HF_REPO_ID is the collection repo, HF_CONFIG_NAME the specific
# dataset (config) inside it. Leave HF_REPO_ID empty to skip the upload.
HF_REPO_ID=latency-sensitive-bench/flappy_200ep
HF_CONFIG_NAME=flappy_fixed_latency_2_200ep_7k2steps

# ---- paths / knobs (override via env) ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
FILTER_PRESET="${FILTER_PRESET:-contiguous_v1}"   # KV-memory needs contiguous rows

# Teacher checkpoint on the hub and where to download it.
CKPT_HF_REPO="latency-sensitive-bench/factor-sweeps-models"
CKPT_SUBPATH="factor_sweeps/flappy/context_window/train/factor_sweep:flappy:frame_stack:fixed_l2:fs1:obs30:stride1:seed10"
CKPT_DOWNLOAD_ROOT="${CKPT_DOWNLOAD_ROOT:-$REPO_ROOT/playground/factor_sweeps_ckpts}"
CHECKPOINT_ROOT="$CKPT_DOWNLOAD_ROOT/$CKPT_SUBPATH"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/scripts/bash_scripts/rollout_data/exports/flappy_fixed_l2}"

# ---- step 1: download the teacher checkpoint (config.json + checkpoint_p0) ----
if [ ! -f "$CHECKPOINT_ROOT/config.json" ]; then
  echo "[download] $CKPT_HF_REPO :: $CKPT_SUBPATH -> $CKPT_DOWNLOAD_ROOT"
  huggingface-cli download "$CKPT_HF_REPO" \
    --include "$CKPT_SUBPATH/*" \
    --local-dir "$CKPT_DOWNLOAD_ROOT"
else
  echo "[download] reuse existing $CHECKPOINT_ROOT"
fi

# ---- step 2: export teacher rollouts ----
PUSH_ARGS=()
if [ -n "$HF_REPO_ID" ]; then
  PUSH_ARGS=(--push-to-hub --hf-repo-id "$HF_REPO_ID")
  if [ -n "${HF_CONFIG_NAME:-}" ]; then
    PUSH_ARGS+=(--hf-config-name "$HF_CONFIG_NAME")
  fi
fi

"$PYTHON" "$REPO_ROOT/scripts/rollout_data/export_sf_teacher_dataset.py" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --env-name flappy \
  --output-dir "$OUTPUT_DIR" \
  --max-episodes "$MAX_EPISODES" \
  --max-steps-per-episode "$MAX_STEPS" \
  --num-envs "$NUM_ENVS" \
  --replay-num-envs "$REPLAY_NUM_ENVS" \
  --device "$DEVICE" \
  --image-writer-workers "$IMAGE_WRITER_WORKERS" \
  --filter-preset "$FILTER_PRESET" \
  "${PUSH_ARGS[@]}"
