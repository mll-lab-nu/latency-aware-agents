#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON=/mnt/ssd/miniconda3/envs/starvla_rl_games_openvla/bin/python
CONDA_LIB=/mnt/ssd/miniconda3/envs/starvla_rl_games_openvla/lib
LANE="$1"
GPU="$2"

export CUDA_HOME=/mnt/ssd/fanqi-lcb-formal/runtime/miles-rootfs/usr/local/cuda-12.9
export PATH="/mnt/ssd/miniconda3/envs/starvla_rl_games_openvla/bin:${CUDA_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CONDA_LIB}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_LIB}:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=9.0
export TORCH_EXTENSIONS_DIR="${REPO_ROOT}/outputs/tasks/decision_history_temporal_25m_seed42/torch_extensions_cuda129"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/sample-factory:${REPO_ROOT}/third_party/flappy-bird-gymnasium"

set -a
source "${REPO_ROOT}/.env"
set +a
export WANDB_ENTITY=saberrr-zju
export WANDB_MODE=online
export WANDB_RESUME=allow

cd "${REPO_ROOT}"

run_job() {
  local env_name="$1"
  local label="$2"
  local family=decision_history_temporal_25m_seed42
  if [[ "${env_name}" == deadly_corridor ]]; then
    family=decision_history_native35_temporal_25m_seed42
  fi
  local manifest="configs/train/${env_name}/${family}/manifest.tsv"
  local run_root="outputs/tasks/${family}/${env_name}/${label}"
  mkdir -p "${run_root}"
  echo "[$(date -Is)] train ${env_name} ${label} on GPU ${GPU}"
  "${PYTHON}" -m scripts.experiment_runner.run_manifest_jobs \
    --manifest "${manifest}" --stage train --gpus "${GPU}" --match "${label}" \
    > "${run_root}/train_runner.log" 2>&1
  echo "[$(date -Is)] best eval ${env_name} ${label} on GPU ${GPU}"
  "${PYTHON}" -m scripts.experiment_runner.run_manifest_jobs \
    --manifest "${manifest}" --stage eval --gpus "${GPU}" --match "${label}" \
    > "${run_root}/best_eval_runner.log" 2>&1
}

if [[ "${LANE}" == deadly ]]; then
  run_job deadly_corridor qwenoft_rtx3090_temporal_hist8_capacity1_native35
  run_job deadly_corridor gr00t_rtx3090_temporal_hist8_capacity1_native35
  run_job deadly_corridor pi05_rtx3090_temporal_hist8_capacity1_native35
elif [[ "${LANE}" == flappy_demon ]]; then
  run_job flappy qwenoft_rtx3090_temporal_hist8_capacity1
  run_job flappy gr00t_rtx3090_temporal_hist8_capacity1
  run_job flappy pi05_rtx3090_temporal_hist8_capacity1
  run_job demon_attack qwenoft_rtx3090_temporal_hist8_capacity1
  run_job demon_attack gr00t_rtx3090_temporal_hist8_capacity1
  run_job demon_attack pi05_rtx3090_temporal_hist8_capacity1
else
  echo "Unknown H100 lane: ${LANE}" >&2
  exit 2
fi

echo "[$(date -Is)] lane ${LANE} complete"
