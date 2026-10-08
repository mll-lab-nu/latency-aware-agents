#!/usr/bin/env bash
set -euo pipefail

# launch_train.py uses paths relative to the StarVLA repository root.
cd "$(dirname "$0")/../.."

# Train the top eight decoder layers and action head.
# The vision stack, tied embeddings, and decoder layers 0-27 remain frozen.
# The run ID remains unchanged to match the published checkpoint artifact.
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=flappy_fix_latency_2_200ep_last_8_layers_corrected_v2 \
  paths.dataset_local_dir=data/flappy_fix_latency_2_200ep \
  trainer.distributed_backend=none \
  trainer.gradient_accumulation_steps=2 \
  trainer.freeze_vit=true \
  trainer.freeze_tied_embedding=true \
  trainer.freeze_llm_layers=[0,27] \
  datasets.vla_data.per_device_batch_size=64 \
  datasets.vla_data.image_mode=single \
  datasets.vla_data.num_obs_frames=1 \
  trainer.max_train_steps=5000 \
  trainer.save_interval=500 \
  rl_games.env_eval.mid_train.enabled=true \
  rl_games.env_eval.mid_train.latencies=[2] \
  rl_games.env_eval.mid_train.interval_steps=250 \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_safetensors_file=true \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
