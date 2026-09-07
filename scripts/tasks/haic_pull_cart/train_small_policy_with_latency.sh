#!/usr/bin/env bash
# Train the native small policy with simulated latency.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEACHER_CHECKPOINT="${1:?usage: $0 TEACHER_CHECKPOINT [LATENCY_CONFIG] [extra Hydra overrides ...]}"
LATENCY_CONFIG="${2:-${REPO_ROOT}/configs/training/latency/haic.yaml}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift
fi
SESSION_PYTHON="${HAIC_PYTHON:-python}"
TEACHER_CHECKPOINT="$(realpath "${TEACHER_CHECKPOINT}")"
LATENCY_CONFIG="$(realpath "${LATENCY_CONFIG}")"

cd "${REPO_ROOT}/third_party/HAIC"
exec "${SESSION_PYTHON}" scripts/train.py \
  algo=ppo_haic_latency_command \
  task=G1/haic/pull_cart \
  task.latency_command=true \
  task.latency_config_path="${LATENCY_CONFIG}" \
  task.latency_command_horizon=1 task.latency_control_repeat=1 \
  task.sim.step_dt=0.05 task.sim.isaac_physics_dt=0.005 task.max_episode_length=400 \
  algo.command_horizon=1 \
  task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0 \
  task.num_envs=1024 total_frames=500000000 \
  checkpoint_path="${TEACHER_CHECKPOINT}" \
  wandb.mode=online "$@"
