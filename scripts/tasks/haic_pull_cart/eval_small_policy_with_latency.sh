#!/usr/bin/env bash
# Evaluate the native small policy with simulated latency.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEACHER_OVERRIDE=()
if [[ "${1:-}" == "--teacher" ]]; then
  TEACHER_OVERRIDE=(algo.eval_teacher=true)
  shift
fi
CHECKPOINT="${1:?usage: $0 [--teacher] CHECKPOINT [LATENCY_CONFIG] [extra Hydra overrides ...]}"
LATENCY_CONFIG="${2:-${REPO_ROOT}/configs/training/latency/haic.yaml}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift
fi
SESSION_PYTHON="${HAIC_PYTHON:-python}"
CHECKPOINT="$(realpath "${CHECKPOINT}")"
LATENCY_CONFIG="$(realpath "${LATENCY_CONFIG}")"

cd "${REPO_ROOT}/third_party/HAIC"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/HAIC${PYTHONPATH:+:${PYTHONPATH}}"
exec "${SESSION_PYTHON}" scripts/eval.py \
  algo=ppo_haic_latency_command \
  task=G1/haic/pull_cart \
  task.latency_command=true \
  task.latency_config_path="${LATENCY_CONFIG}" \
  task.latency_command_horizon=1 task.latency_control_repeat=1 \
  task.sim.step_dt=0.05 task.sim.isaac_physics_dt=0.005 task.max_episode_length=400 \
  algo.command_horizon=1 task.num_envs=100 \
  task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0 \
  "${TEACHER_OVERRIDE[@]}" \
  checkpoint_path="${CHECKPOINT}" \
  "$@"
