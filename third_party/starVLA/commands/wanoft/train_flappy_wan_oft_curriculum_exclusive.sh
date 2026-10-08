#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash commands/wanoft/train_flappy_wan_oft_curriculum_exclusive.sh
#   bash commands/wanoft/train_flappy_wan_oft_curriculum_exclusive.sh trainer.max_train_steps=1000

CONTEXT_WINDOW="${CONTEXT_WINDOW:-5}"
EPISODES_PER_LATENCY="${EPISODES_PER_LATENCY:-200}"
LATENCY_FILTER="${LATENCY_FILTER:-[0,1,2,3,4]}"
PHASE_STEPS="${PHASE_STEPS:-null}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))}"
POST_TRAIN_EPISODES="${POST_TRAIN_EPISODES:-20}"
POST_TRAIN_LATENCIES="${POST_TRAIN_LATENCIES:-[0,1,2,3,4]}"
SAVE_AT_PHASE_END="${SAVE_AT_PHASE_END:-false}"
DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/flappy_mixed_latency_${EPISODES_PER_LATENCY}ep_per_lat_context${CONTEXT_WINDOW}}"
RUN_ID="${RUN_ID:-wan_oft_flappy_mixed_latency_curriculum_exclusive_context${CONTEXT_WINDOW}_${EPISODES_PER_LATENCY}ep_per_lat_standard_sft_${MAX_TRAIN_STEPS}_effbs${EFFECTIVE_BATCH_SIZE}_224_currentce}"
PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-${DATASET_LOCAL_DIR}/flappy_mixed_latency_train__bridge/latency_prompt_map.json}"

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

python examples/rl_games/scripts/launch_train.py \
  model=wan_oft \
  env=flappy \
  init=wan_oft_libero \
  mode=curriculum_exclusive \
  run_id="${RUN_ID}" \
  paths.dataset_local_dir="${DATASET_LOCAL_DIR}" \
  dataset.converted_name=flappy_mixed_latency_train__bridge \
  "dataset.latency_filter=${LATENCY_FILTER}" \
  dataset.episodes_per_latency="${EPISODES_PER_LATENCY}" \
  trainer.distributed_backend=none \
  launch.use_accelerate=false \
  launch.num_processes=1 \
  trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  datasets.vla_data.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  datasets.vla_data.data_mix=flappy_mixed_latency_train__bridge \
  "datasets.vla_data.obs_image_size=[224,224]" \
  datasets.vla_data.image_sequence_length="${CONTEXT_WINDOW}" \
  "datasets.vla_data.observation_indices=[-4,-3,-2,-1,0]" \
  datasets.vla_data.sequential_step_sampling=true \
  datasets.vla_data.action_balance.enabled=false \
  datasets.vla_data.latency_curriculum.enabled=true \
  datasets.vla_data.latency_curriculum.strategy=exclusive \
  "datasets.vla_data.latency_curriculum.latencies=${LATENCY_FILTER}" \
  "datasets.vla_data.latency_curriculum.phase_steps=${PHASE_STEPS}" \
  datasets.vla_data.latency_curriculum.eval_at_phase_end=false \
  datasets.vla_data.latency_curriculum.save_at_phase_end="${SAVE_AT_PHASE_END}" \
  framework.world_model.num_frames="${CONTEXT_WINDOW}" \
  framework.action_model.loss_type=current_discrete_ce \
  trainer.learning_rate.action_query_proj=1.0e-4 \
  trainer.max_train_steps="${MAX_TRAIN_STEPS}" \
  trainer.num_warmup_steps=0 \
  trainer.lr_scheduler_type=cosine_with_min_lr \
  trainer.scheduler_specific_kwargs.min_lr=1.0e-6 \
  trainer.save_interval=0 \
  rl_games.env_eval.enabled=true \
  rl_games.env_eval.eval_backend=eval_core \
  rl_games.env_eval.image_size=224 \
  rl_games.env_eval.vectorized.enabled=false \
  rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}" \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=true \
  "rl_games.env_eval.post_train.latencies=${POST_TRAIN_LATENCIES}" \
  rl_games.env_eval.post_train.num_episodes="${POST_TRAIN_EPISODES}" \
  rl_games.env_eval.post_train.max_steps_per_episode=3600 \
  checkpoint.save_pt_file=true \
  checkpoint.save_training_state=false \
  checkpoint.save_best_model=false \
  checkpoint.save_final_model=true \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
