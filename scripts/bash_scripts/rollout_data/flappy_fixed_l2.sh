#!/usr/bin/env bash
# Export a Flappy teacher rollout dataset for KV-memory training.
# Step 1: download the factor-sweep teacher checkpoint from the Hugging Face hub.
# Step 2: replay it and export canonical rollout rows at the observation cadence.
set -euo pipefail
export HF_ENDPOINT=https://huggingface.co

# ---- run variables ----
MAX_EPISODES=200
MAX_STEPS=7200
NUM_ENVS="${NUM_ENVS:-4096}"
REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS:-32}"
DEVICE=gpu
IMAGE_WRITER_WORKERS=32
# HF dataset target: HF_REPO_ID is the collection repo, HF_CONFIG_NAME the specific
# dataset (config) inside it. Leave HF_REPO_ID empty to skip the upload.
HF_REPO_ID=MLL-Lab/LAGEN-datasets
HF_CONFIG_NAME=flappy_fixed_latency_2_200ep_7k2steps

# ---- paths / knobs (override via env) ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHON="${PYTHON:-python}"
FILTER_PRESET="${FILTER_PRESET:-contiguous_v1}"   # KV-memory needs contiguous rows

# Fixed canonical checkpoint for the published context-window teacher.
CKPT_HF_REPO="MLL-Lab/LAGEN-models"
CKPT_HF_REVISION="365fe5402de3fb464d5da8da9f944b969c5801c4"
CKPT_SUBPATH="latency-aware/flappy/small-policy/factor_sweep:flappy:frame_stack:fixed_l2:fs1:obs30:stride1:seed10"
CKPT_DOWNLOAD_ROOT="${CKPT_DOWNLOAD_ROOT:-$REPO_ROOT/playground/published_teacher_ckpts}"
CKPT_SNAPSHOT_ROOT="$CKPT_DOWNLOAD_ROOT/$CKPT_HF_REVISION"
CHECKPOINT_ROOT="$CKPT_SNAPSHOT_ROOT/$CKPT_SUBPATH"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/scripts/bash_scripts/rollout_data/exports/flappy_fixed_l2}"

# ---- step 1: download the teacher checkpoint (config.json + checkpoint_p0) ----
huggingface-cli download "$CKPT_HF_REPO" \
  --revision "$CKPT_HF_REVISION" \
  --include "$CKPT_SUBPATH/config.json" "$CKPT_SUBPATH/checkpoint_p0/*" \
  --local-dir "$CKPT_SNAPSHOT_ROOT"

# ---- step 2: export teacher rollouts ----
MAX_EPISODES="$MAX_EPISODES" MAX_STEPS="$MAX_STEPS" NUM_ENVS="$NUM_ENVS" \
REPLAY_NUM_ENVS="$REPLAY_NUM_ENVS" DEVICE="$DEVICE" IMAGE_WRITER_WORKERS="$IMAGE_WRITER_WORKERS" \
FILTER_PRESET="$FILTER_PRESET" VAL_FRACTION=0.1 HF_REPO_ID="$HF_REPO_ID" HF_CONFIG_NAME="$HF_CONFIG_NAME" \
bash "$REPO_ROOT/scripts/bash_scripts/run_sf_teacher_dataset_export.sh" "$CHECKPOINT_ROOT" "$OUTPUT_DIR"
