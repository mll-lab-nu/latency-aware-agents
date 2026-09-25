#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model}"
TRAIN_DEVICES="${2:?two GPU ids}"
SIMULATOR_GPU="${3:?simulator GPU id}"
PORT="${4:?port}"
OUTPUT="${5:?output path}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="${ROOT}/outputs/starvla"
SOURCE="${BASE}/runs/mikasa_intercept_grab_fast/h8-zero-latency/${MODEL}"
BOOTSTRAP="${BASE}/datasets/mikasa_intercept_grab_fast/teacher-rollouts-h8-state-v1"
TEACHER="${BASE}/artifacts/models/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt"
TASK_PY="${ROOT}/third_party/MIKASA-Robo/.venv/bin/python"
INFERENCE_GPU="${TRAIN_DEVICES%%,*}"

case "${MODEL}" in
  qwenoft) VLA_PY=/path/to/user/miniconda3/envs/starvla_rl_games_openvla/bin/python ;;
  qwenpi_v3) VLA_PY=/path/to/user/miniconda3/envs/starvla_rl_games_pi05/bin/python ;;
  qwengr00t) VLA_PY=/path/to/user/miniconda3/envs/starvla_rl_games_gr00t/bin/python ;;
  *) exit 2 ;;
esac

cd "${ROOT}"
set -a
. ./.env
set +a
export WANDB_ENTITY=placeholder
export PYTHONPATH="${ROOT}:${ROOT}/third_party/starVLA"
UV_EXEC="${UV:-/path/to/user/.local/bin/uv}"

exec "${VLA_PY}" -m scripts.tasks.mikasa_intercept_grab_fast.run_dagger \
  --model "${MODEL}" --condition zero-latency \
  --source-run "${SOURCE}" --bootstrap "${BOOTSTRAP}" \
  --eval-config "${SOURCE}/evals/final-zero/eval_config.yaml" \
  --eval-config-profile "${SOURCE}/evals/final-profile-no-startup-wait/eval_config.yaml" \
  --eval-config-fixed "${SOURCE}/evals/final-fixed-2/eval_config.yaml" \
  --latency-config "${SOURCE}/evals/final-zero/eval_config.yaml" \
  --teacher-checkpoint "${TEACHER}" --output "${OUTPUT}" \
  --vla-python "${VLA_PY}" --task-python "${TASK_PY}" --uv "${UV_EXEC}" \
  --train-devices "${TRAIN_DEVICES}" --inference-gpu "${INFERENCE_GPU}" \
  --simulator-gpu "${SIMULATOR_GPU}" \
  --gpu-lock "/tmp/mikasa-h8-dagger-gpu${SIMULATOR_GPU}.lock" \
  --port "${PORT}"
