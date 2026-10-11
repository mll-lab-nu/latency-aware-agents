#!/usr/bin/env bash
# Sharded Flappy teacher-rollout export for KV-memory training.
# Speeds up the PROBE phase (the bottleneck: stepping millions of agent-steps to find the
# rare long episodes) by running K probe-only workers that share the idle GPU and explore
# disjoint reset-seed ranges. Their accepted specs are merged into one resume state, then
# the existing single-process REPLAY materializes the dataset (images/rows/parquet) and
# pushes to the hub. Replay is intentionally single-process so its byte-for-byte behavior
# is unchanged from flappy_fixed_l2.sh.
set -euo pipefail
export HF_ENDPOINT=https://huggingface.co

# ---- run variables ----
MAX_EPISODES="${MAX_EPISODES:-200}"
MAX_STEPS="${MAX_STEPS:-7200}"
NUM_SHARDS="${NUM_SHARDS:-6}"
# Per-shard probe batch. Throughput is ~flat across batch size (per-core bound), so a small
# batch keeps each worker's GPU memory low enough for K workers to share one GPU.
SHARD_NUM_ENVS="${SHARD_NUM_ENVS:-256}"
REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS:-16}"
DEVICE="${DEVICE:-gpu}"
IMAGE_WRITER_WORKERS="${IMAGE_WRITER_WORKERS:-32}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-0}"
FILTER_PRESET="${FILTER_PRESET:-contiguous_v1}"   # KV-memory needs contiguous rows
# Disjoint seed namespace per shard: shard k uses base_seed offset k*STRIDE. STRIDE must
# exceed the max attempts any single shard makes (accept rate is low, so this is generous).
SEED_STRIDE="${SEED_STRIDE:-100000000}"

# HF dataset target (collection repo + config). Leave HF_REPO_ID empty to skip upload.
HF_REPO_ID="${HF_REPO_ID:-MLL-Lab/LAGEN-datasets}"
HF_CONFIG_NAME="${HF_CONFIG_NAME:-flappy_fixed_latency_2_200ep_7k2steps}"

# ---- paths / knobs ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHON="${PYTHON:-python}"
EXPORT_RECIPE="$REPO_ROOT/scripts/bash_scripts/run_sf_teacher_dataset_export.sh"
MERGE_PY="$REPO_ROOT/scripts/rollout_data/merge_probe_shards.py"

# Fixed canonical checkpoint for the published context-window teacher.
CKPT_HF_REPO="MLL-Lab/LAGEN-models"
CKPT_HF_REVISION="365fe5402de3fb464d5da8da9f944b969c5801c4"
CKPT_SUBPATH="latency-aware/flappy/small-policy/factor_sweep:flappy:frame_stack:fixed_l2:fs1:obs30:stride1:seed10"
CKPT_DOWNLOAD_ROOT="${CKPT_DOWNLOAD_ROOT:-$REPO_ROOT/playground/published_teacher_ckpts}"
CKPT_SNAPSHOT_ROOT="$CKPT_DOWNLOAD_ROOT/$CKPT_HF_REVISION"
CHECKPOINT_ROOT="$CKPT_SNAPSHOT_ROOT/$CKPT_SUBPATH"

RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/scripts/bash_scripts/rollout_data/exports/flappy_fixed_l2_sharded}"
SHARD_ROOT="$RUN_ROOT/shards"
FINAL_DIR="$RUN_ROOT/final"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$SHARD_ROOT" "$FINAL_DIR" "$LOG_DIR"

# Per-shard quota with a small margin so the K shards together comfortably exceed
# MAX_EPISODES even if one seed range yields slightly fewer long episodes.
SHARD_EPISODES="${SHARD_EPISODES:-$(( (MAX_EPISODES + NUM_SHARDS - 1) / NUM_SHARDS + 5 ))}"

echo "[sharded] shards=$NUM_SHARDS shard_episodes=$SHARD_EPISODES shard_num_envs=$SHARD_NUM_ENVS max_episodes=$MAX_EPISODES"

# ---- step 1: download teacher checkpoint ----
huggingface-cli download "$CKPT_HF_REPO" \
  --revision "$CKPT_HF_REVISION" \
  --include "$CKPT_SUBPATH/config.json" "$CKPT_SUBPATH/checkpoint_p0/*" \
  --local-dir "$CKPT_SNAPSHOT_ROOT"

# ---- step 2: launch K probe-only workers (disjoint seeds, shared GPU) ----
SHARD_DIRS=()
pids=()
for ((k=0; k<NUM_SHARDS; k++)); do
  shard_dir="$SHARD_ROOT/shard_$k"
  SHARD_DIRS+=("$shard_dir")
  offset=$(( k * SEED_STRIDE ))
  log="$LOG_DIR/probe_shard_$k.log"
  echo "[probe] shard $k offset=$offset -> $shard_dir (log: $log)"
  MAX_EPISODES="$SHARD_EPISODES" MAX_STEPS="$MAX_STEPS" NUM_ENVS="$SHARD_NUM_ENVS" \
  DEVICE="$DEVICE" VAL_FRACTION="$VAL_FRACTION" SPLIT_SEED="$SPLIT_SEED" \
  FILTER_PRESET="$FILTER_PRESET" ROLLOUT_SEED_OFFSET="$offset" PROBE_ONLY=1 \
  bash "$EXPORT_RECIPE" "$CHECKPOINT_ROOT" "$shard_dir" > "$log" 2>&1 &
  pids+=("$!")
done

fail=0
for ((j=0; j<${#pids[@]}; j++)); do
  if wait "${pids[$j]}"; then
    echo "[probe] shard $j OK"
  else
    echo "[probe] shard $j FAILED (see $LOG_DIR/probe_shard_$j.log)" >&2
    fail=1
  fi
done
[ "$fail" -eq 0 ] || { echo "[sharded] a probe shard failed; aborting" >&2; exit 1; }

# ---- step 3: merge shard probe states into the final resume state ----
MERGE_ARGS=()
for d in "${SHARD_DIRS[@]}"; do MERGE_ARGS+=(--shard-dir "$d"); done
"$PYTHON" "$MERGE_PY" \
  "${MERGE_ARGS[@]}" \
  --output-dir "$FINAL_DIR" \
  --max-episodes "$MAX_EPISODES" \
  --val-fraction "$VAL_FRACTION" \
  --split-seed "$SPLIT_SEED"

# ---- step 4: single-process replay from the merged state (skips probe), then push ----
MAX_EPISODES="$MAX_EPISODES" MAX_STEPS="$MAX_STEPS" NUM_ENVS="$SHARD_NUM_ENVS" \
REPLAY_NUM_ENVS="$REPLAY_NUM_ENVS" DEVICE="$DEVICE" IMAGE_WRITER_WORKERS="$IMAGE_WRITER_WORKERS" \
VAL_FRACTION="$VAL_FRACTION" SPLIT_SEED="$SPLIT_SEED" FILTER_PRESET="$FILTER_PRESET" \
HF_REPO_ID="$HF_REPO_ID" HF_CONFIG_NAME="$HF_CONFIG_NAME" \
bash "$EXPORT_RECIPE" "$CHECKPOINT_ROOT" "$FINAL_DIR"

echo "[sharded] done -> $FINAL_DIR"
