#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="$(command -v "${FLAPPY_PYTHON:-python}")"
HF_CLI="${FLAPPY_HF_CLI:-hf}"
PROFILE_REVISION="48d13d3dfa2b77c2d10fbbbcc5210d8af9fc8749"
SPEC="configs/experiments/flappy/small_policy_qwen_profiles_10fps_25m.yaml"
MANIFEST="configs/train/flappy/profiled_small_qwen_10fps_25m/manifest.tsv"

export CUDA_HOME=/mnt/ssd/fanqi-lcb-formal/runtime/miles-rootfs/usr/local/cuda-12.9
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR="${REPO_ROOT}/outputs/tasks/flappy/profiled_small_qwen_10fps_25m/torch_extensions_cuda129"

set -a
source "${REPO_ROOT}/.env"
set +a
export WANDB_ENTITY=saberrr-zju
export WANDB_MODE=online
export WANDB_RESUME=allow

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -c \
  "from training.flappy_sf.flappy_cuda_extension import load_flappy_cuda_extension; load_flappy_cuda_extension()"

"${HF_CLI}" download latency-sensitive-bench/profiles \
  --repo-type dataset \
  --revision "${PROFILE_REVISION}" \
  --include "profiles/openvla/1x-rtx3090/flappy/instance_a5037b165aa0cedc/*" \
  "profiles/gr00t/1x-rtx3090/flappy/instance_a5037b165aa0cedc/*" \
  "profiles/pi05/1x-rtx3090/flappy/instance_a5037b165aa0cedc/*" \
  --local-dir results/profiling

"${PYTHON}" scripts/sweeps/generate_sweep_configs.py \
  --spec "${SPEC}" \
  --eval-checkpoint-kind best

"${PYTHON}" -m scripts.experiment_runner.run_manifest_jobs \
  --manifest "${MANIFEST}" \
  --stage train \
  --gpus 0,1

exec "${PYTHON}" -m scripts.experiment_runner.run_manifest_jobs \
  --manifest "${MANIFEST}" \
  --stage eval \
  --gpus 0,1
