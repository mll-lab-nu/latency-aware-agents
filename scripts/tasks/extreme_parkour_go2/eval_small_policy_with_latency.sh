#!/usr/bin/env bash
# Evaluate the native small policy with simulated latency.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CHECKPOINT="${1:?usage: $0 CHECKPOINT [LATENCY_CONFIG TEACHER_CHECKPOINT] [extra eval args ...]}"
shift

PYTHON="${PARKOUR_PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
COMMAND=(
  "${PYTHON}" "${REPO_ROOT}/third_party/Isaaclab_Parkour/scripts/rsl_rl/evaluation.py"
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0
  --checkpoint "$(realpath "${CHECKPOINT}")"
  --num_envs 256
  --eval_episodes 100
  --max_episode_steps 1500
  --headless
)
if [[ $# -gt 0 && "${1}" != --* ]]; then
  LATENCY_CONFIG="$(realpath "${1}")"
  TEACHER_CHECKPOINT="${2:?a teacher checkpoint is required with LATENCY_CONFIG}"
  shift 2
  COMMAND+=(
    --latency-config "${LATENCY_CONFIG}"
    --latency-teacher-checkpoint "$(realpath "${TEACHER_CHECKPOINT}")"
  )
fi

cd "${REPO_ROOT}/third_party/Isaaclab_Parkour"
exec "${COMMAND[@]}" "$@"
