#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/outputs/haic/phase1_teacher"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}/third_party/HAIC"
export WANDB_NAME=haic-phase1-teacher
export WANDB_DIR="${OUTPUT_ROOT}"
exec python scripts/train.py \
  algo=ppo_haic_train task=G1/haic/pull_cart \
  task.num_envs=4096 total_frames=500000000 seed=0 \
  task.action.min_delay=0 task.action.max_delay=0 task.action.alpha=1.0 \
  wandb.mode=online hydra.run.dir="${OUTPUT_ROOT}" "$@"
