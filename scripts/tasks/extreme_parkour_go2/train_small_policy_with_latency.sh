#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEACHER_CHECKPOINT="${1:?usage: $0 TEACHER_CHECKPOINT [LATENCY_CONFIG] [extra train args ...]}"
LATENCY_CONFIG="${REPO_ROOT}/configs/training/latency/parkour.yaml"
PYTHON="${PARKOUR_PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ $# -ge 2 && "${2}" != --* ]]; then
  LATENCY_CONFIG="${2}"
  shift 2
else
  shift
fi
TEACHER_CHECKPOINT="$(realpath "${TEACHER_CHECKPOINT}")"
LATENCY_CONFIG="$(realpath "${LATENCY_CONFIG}")"

cd "${REPO_ROOT}/third_party/Isaaclab_Parkour"
exec "${PYTHON}" scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
  --latency-config "${LATENCY_CONFIG}" \
  --latency-teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --headless --seed 1 --num_envs 1024 --max_iterations 15000 \
  --run_name extreme_parkour_go2_ppo_small_policy_with_latency_seed1_15k \
  --logger wandb --log_project_name isaaclab-parkour "$@"
