#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}/third_party/Isaaclab_Parkour"
exec env LD_PRELOAD=/lib/x86_64-linux-gnu/libnccl.so.2 \
  NCCL_DEBUG=WARN WANDB_MODE=online OMNI_KIT_ACCEPT_EULA=YES \
  python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --master_port=29542 \
  scripts/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 \
  --distributed --headless --seed 1 \
  --num_envs 3072 --max_iterations 15000 \
  --run_name teacher_zero_delay_fixed_lr_ddp_seed1_15k \
  --logger wandb --log_project_name isaaclab-parkour "$@"
