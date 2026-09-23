#!/usr/bin/env bash
set -euo pipefail

TASK="$1"
MODEL="$2"
GPU="$3"
RUN_ID="$4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

case "${MODEL}" in
  qwenoft) PROFILE_FAMILY=openvla ;;
  qwengr00t) PROFILE_FAMILY=gr00t ;;
  qwenpi_v3) PROFILE_FAMILY=pi05 ;;
esac

set -a
. "${REPO_ROOT}/.wandb.env"
set +a
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/starVLA:${REPO_ROOT}/third_party/sample-factory:${REPO_ROOT}/third_party/flappy-bird-gymnasium:${REPO_ROOT}/third_party/cule"

VLA_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/python
exec "${VLA_PY}" -m scripts.tasks.decision_history.eval_vla_hist8 eval \
  --task "${TASK}" --model "${MODEL}" \
  --bundle-dir "${REPO_ROOT}/outputs/tasks/decision_history_vla_hist8_v2/bundles/${RUN_ID}" \
  --profile-path "${REPO_ROOT}/results/profiling/profiles/${PROFILE_FAMILY}/1x-rtx3090/${TASK}/instance_a5037b165aa0cedc/profile.json" \
  --backbone-path "${REPO_ROOT}/backbone/Qwen3-VL-4B-Instruct" \
  --vla-python "${VLA_PY}" \
  --output-dir "${REPO_ROOT}/outputs/tasks/decision_history_vla_hist8_v2/evals/${RUN_ID}" \
  --device cuda:0 --episodes 100
