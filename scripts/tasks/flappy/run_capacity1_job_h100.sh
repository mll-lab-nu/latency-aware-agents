#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON=/mnt/ssd/miniconda3/envs/starvla_rl_games_openvla/bin/python
GPU="$1"
MODEL="$2"
STAGE="$3"
MANIFEST="$4"

export CUDA_HOME=/mnt/ssd/fanqi-lcb-formal/runtime/miles-rootfs/usr/local/cuda-12.9
export PATH="/mnt/ssd/miniconda3/envs/starvla_rl_games_openvla/bin:${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR="${REPO_ROOT}/outputs/tasks/flappy/torch_extensions_cuda129/${MODEL}"

set -a
source "${REPO_ROOT}/.env"
set +a
export WANDB_ENTITY=saberrr-zju
export WANDB_MODE=online
export WANDB_RESUME=allow

cd "${REPO_ROOT}"
exec "${PYTHON}" -m scripts.experiment_runner.run_manifest_jobs \
  --manifest "${MANIFEST}" --stage "${STAGE}" --gpus "${GPU}" --match "${MODEL}"
