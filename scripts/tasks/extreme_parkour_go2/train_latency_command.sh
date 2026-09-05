#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEACHER_CHECKPOINT="${1:?usage: $0 TEACHER_CHECKPOINT [LATENCY_CONFIG] [extra train args ...]}"
LATENCY_CONFIG="${2:-${REPO_ROOT}/configs/training/latency/parkour.yaml}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift
fi
TEACHER_CHECKPOINT="$(realpath "${TEACHER_CHECKPOINT}")"
LATENCY_CONFIG="$(realpath "${LATENCY_CONFIG}")"

cd "${REPO_ROOT}/third_party/Isaaclab_Parkour"
exec python scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
  --latency-config "${LATENCY_CONFIG}" \
  --latency-teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --headless --seed 1 --num_envs 1024 --max_iterations 15000 \
  --run_name latency_command_seed1_15k "$@"
