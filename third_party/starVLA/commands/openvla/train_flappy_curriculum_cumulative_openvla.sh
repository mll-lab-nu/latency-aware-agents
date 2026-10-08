#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash commands/openvla/train_flappy_curriculum_cumulative_openvla.sh

LATENCY_FILTER="${LATENCY_FILTER:-[0,1,2,3,4]}"
EPISODES_PER_LATENCY="${EPISODES_PER_LATENCY:-40}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/flappy_mixed_latency_${EPISODES_PER_LATENCY}ep_per_lat}"
RUN_ID="${RUN_ID:-flappy_curriculum_cumulative_${EPISODES_PER_LATENCY}ep_per_latency}"
PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-prompt/flappy_latency_prompt_map.json}"

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  mode=curriculum_cumulative \
  run_id="${RUN_ID}" \
  paths.dataset_local_dir="${DATASET_LOCAL_DIR}" \
  "dataset.latency_filter=${LATENCY_FILTER}" \
  dataset.episodes_per_latency="${EPISODES_PER_LATENCY}" \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  datasets.vla_data.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  trainer.max_train_steps="${MAX_TRAIN_STEPS}" \
  trainer.save_interval="${SAVE_INTERVAL}" \
  rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}" \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
