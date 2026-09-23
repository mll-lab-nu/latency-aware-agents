#!/usr/bin/env bash
set -euo pipefail

TASK="$1"
MODEL="$2"
GPU_PAIR="$3"
PORT="$4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

case "${MODEL}" in
  qwenoft) VLA_ENV=starvla_rl_games_openvla ;;
  qwengr00t) VLA_ENV=starvla_rl_games_gr00t ;;
  qwenpi_v3) VLA_ENV=starvla_rl_games_pi05 ;;
esac

set -a
. "${REPO_ROOT}/.wandb.env"
set +a
export WANDB_ENTITY=saberrr-zju
export CUDA_VISIBLE_DEVICES="${GPU_PAIR}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/starVLA:${REPO_ROOT}/third_party/sample-factory"

exec "/home/ubuntu/miniconda3/envs/${VLA_ENV}/bin/python" \
  -m scripts.tasks.decision_history.train_vla_hist8 \
  --task "${TASK}" --model "${MODEL}" \
  --dataset-dir "${REPO_ROOT}/outputs/tasks/decision_history_vla_hist8/data/${TASK}/${MODEL}/lerobot" \
  --base-model-dir "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17" \
  --output-root "${REPO_ROOT}/outputs/tasks/decision_history_vla_hist8_v2/train" \
  --port "${PORT}" "${@:5}"
