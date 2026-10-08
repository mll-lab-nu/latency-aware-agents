#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# Streaming-KV full tuning with serial replay. The dataloader emits an
# eight-frame rollout with one action label per frame. Training performs eight
# sequential forwards through a four-frame cache and detaches cached states
# between frames, giving truncated BPTT. The rollout exceeds the cache window
# so training exercises eviction.
#
# At eval the memory is read from the checkpoint config automatically: only the newest
# frame is encoded each step (past frames come from the per-slot cache), so the eval env
# can keep frame_stack=1. The per-slot memory is reset at episode boundaries.

# DeepSpeed ZeRO-2 runs on two processes. Effective batch:
# per_device_batch_size * num_processes * gradient_accumulation_steps
# = 4 * 2 * 4 = 32 rollouts, or 256 per-frame action labels, per optimizer step.
python examples/rl_games/scripts/launch_train.py \
  model=openvla \
  env=flappy \
  init=bridge \
  run_id=flappy_fix_latency_2_200ep_7k2steps_kv_memory \
  paths.dataset_local_dir=data/flappy_fix_latency_2_200ep_7k2steps \
  datasets.vla_data.num_obs_frames=8 \
  datasets.vla_data.image_mode=multiframe \
  framework.qwenvl.attn_implementation=flash_attention_2 \
  framework.qwenvl.enable_gradient_checkpointing=false \
  framework.kv_memory.enabled=true \
  framework.kv_memory.window=4 \
  framework.kv_memory.rollout_len=8 \
  trainer.distributed_backend=deepspeed \
  trainer.eval_interval=250 \
  trainer.eval_num_batches=50 \
  trainer.eval_action_classification=false \
  launch.use_accelerate=true \
  launch.num_processes=2 \
  paths.accelerate_config=starVLA/config/deepseeds/deepspeed_zero2.yaml \
  trainer.gradient_accumulation_steps=4 \
  datasets.vla_data.per_device_batch_size=4 \
  trainer.max_train_steps=4000 \
  trainer.save_interval=500 \
  rl_games.env_eval.eval_parallel_envs=5 \
  rl_games.env_eval.distributed_mode=rank_sharded \
  rl_games.env_eval.mid_train.enabled=true \
  rl_games.env_eval.mid_train.latencies=[2] \
  rl_games.env_eval.mid_train.interval_steps=250 \
  rl_games.env_eval.mid_train.num_episodes=20 \
  rl_games.env_eval.post_train.enabled=true \
  rl_games.env_eval.post_train.latencies=[2] \
  rl_games.env_eval.post_train.num_episodes=50 \
  checkpoint.save_pt_file=false \
  checkpoint.save_safetensors_file=true \
  checkpoint.save_best_model=false \
  checkpoint.local.keep_last_n=1 \
  checkpoint.load=none
