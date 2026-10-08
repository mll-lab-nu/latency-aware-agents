#!/usr/bin/env bash
# QwenOFT on flappy with TEMPORAL multi-image input (per-frame tokenized).
# num_obs_frames=4 -> the dataloader emits the last 4 frames (oldest..newest) and
# each is tokenized as its own image. Set the eval env's frame_stack to 4 to match.
# Also shows trainer.freeze_llm_layers: freeze the bottom 50% of Qwen3-VL LLM layers.

# export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=flappy_qwenoft_multiframe4 \
  paths.dataset_local_dir=data/flappy_fix_latency_0_200ep \
  datasets.vla_data.num_obs_frames=4 \
  datasets.vla_data.image_mode=multiframe \
  trainer.freeze_vit=false \
  trainer.distributed_backend=deepspeed \
  trainer.gradient_accumulation_steps=4 \
  datasets.vla_data.per_device_batch_size=16 \
  trainer.max_train_steps=4000 \
  trainer.save_interval=500 \
  rl_games.env_eval.mid_train.enabled=false \
  rl_games.env_eval.post_train.enabled=false \
  checkpoint.save_pt_file=false \
  checkpoint.save_safetensors_file=true \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
