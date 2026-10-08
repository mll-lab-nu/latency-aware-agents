#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   bash commands/wanoft/train_asterix_wan_oft.sh
#   LATENCY=0 bash commands/wanoft/train_asterix_wan_oft.sh
LATENCY="${1:-${LATENCY:-0}}"
if [[ $# -gt 0 ]]; then
  shift
fi
if ! [[ "${LATENCY}" =~ ^[0-9]+$ ]]; then
  echo "LATENCY must be a non-negative integer, got: ${LATENCY}" >&2
  exit 2
fi

CONTEXT_WINDOW="${CONTEXT_WINDOW:-5}"
MAX_EPISODES="${MAX_EPISODES:-1000}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-7000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))}"
POST_TRAIN_NUM_EPISODES="${POST_TRAIN_NUM_EPISODES:-20}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-3600}"
POST_TRAIN_LATENCIES="${POST_TRAIN_LATENCIES:-[0,2,4,6]}"
DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-data/asterix_fix_latency_${LATENCY}_${MAX_EPISODES}ep_context${CONTEXT_WINDOW}}"
RUN_ID="${RUN_ID:-wan_oft_asterix_factorized6_fix_latency_${LATENCY}_context${CONTEXT_WINDOW}_standard_sft_${MAX_TRAIN_STEPS}_effbs${EFFECTIVE_BATCH_SIZE}_224_currentfactorizedce}"
PROMPT_MAP_PATH="${PROMPT_MAP_PATH:-${DATASET_LOCAL_DIR}/asterix_train__bridge/latency_prompt_map.json}"

OBSERVATION_INDICES="["
for ((offset = CONTEXT_WINDOW - 1; offset >= 1; offset--)); do
  OBSERVATION_INDICES+="-${offset},"
done
OBSERVATION_INDICES+="0]"

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=wan_oft \
  env=asterix \
  init=wan_oft_libero \
  run_id="${RUN_ID}" \
  paths.dataset_local_dir="${DATASET_LOCAL_DIR}" \
  dataset.converted_name=asterix_train__bridge \
  trainer.distributed_backend=none \
  launch.use_accelerate=false \
  launch.num_processes=1 \
  trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  datasets.vla_data.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  datasets.vla_data.data_mix=asterix_train__bridge \
  "datasets.vla_data.obs_image_size=[224,224]" \
  datasets.vla_data.image_sequence_length="${CONTEXT_WINDOW}" \
  "datasets.vla_data.observation_indices=${OBSERVATION_INDICES}" \
  datasets.vla_data.sequential_step_sampling=false \
  datasets.vla_data.action_balance.enabled=false \
  framework.world_model.num_frames="${CONTEXT_WINDOW}" \
  framework.action_model.action_layout=asterix_factorized_6 \
  framework.action_model.loss_type=asterix_factorized_ce \
  trainer.learning_rate.action_query_proj=1.0e-4 \
  trainer.max_train_steps="${MAX_TRAIN_STEPS}" \
  trainer.num_warmup_steps=0 \
  trainer.lr_scheduler_type=cosine_with_min_lr \
  trainer.scheduler_specific_kwargs.min_lr=1.0e-6 \
  trainer.eval_interval="${MAX_TRAIN_STEPS}" \
  trainer.save_interval="${MAX_TRAIN_STEPS}" \
  checkpoint.save_pt_file=true \
  checkpoint.save_training_state=false \
  checkpoint.save_best_model=false \
  checkpoint.save_final_model=true \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none \
  rl_games.env_eval.enabled=true \
  rl_games.env_eval.eval_backend=eval_core \
  rl_games.env_eval.image_size=224 \
  rl_games.env_eval.vectorized.enabled=false \
  rl_games.env_eval.asterix.action_layout=factorized_6 \
  rl_games.env_eval.latency.prompt_map_path="${PROMPT_MAP_PATH}" \
  rl_games.env_eval.mid_train.enabled=true \
  rl_games.env_eval.mid_train.interval_steps=1000 \
  rl_games.env_eval.mid_train.num_episodes=5 \
  rl_games.env_eval.mid_train.max_steps_per_episode="${MAX_STEPS_PER_EPISODE}" \
  rl_games.env_eval.post_train.enabled=true \
  "rl_games.env_eval.post_train.latencies=${POST_TRAIN_LATENCIES}" \
  rl_games.env_eval.post_train.num_episodes="${POST_TRAIN_NUM_EPISODES}" \
  rl_games.env_eval.post_train.max_steps_per_episode="${MAX_STEPS_PER_EPISODE}" \
  "$@"
