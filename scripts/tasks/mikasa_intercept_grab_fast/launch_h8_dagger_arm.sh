#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model}"
CONDITION="${2:?condition}"
TRAIN_DEVICES="${3:?two GPU ids}"
SIMULATOR_GPU="${4:?simulator GPU id}"
PORT="${5:?port}"
OUTPUT="${6:?output path}"
MODE="${7:-formal}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE="${ROOT}/outputs/starvla"
SOURCE="${MIKASA_H8_SOURCE_RUN:?Set MIKASA_H8_SOURCE_RUN to the SFT run}"
DATA="${MIKASA_H8_DATA_ROOT:?Set MIKASA_H8_DATA_ROOT}"
TEACHER="${BASE}/artifacts/models/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt"
TASK_PY="${ROOT}/third_party/MIKASA-Robo/.venv/bin/python"
INFERENCE_GPU="${TRAIN_DEVICES%%,*}"

case "${MODEL}" in
  qwenoft) VLA_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/python ;;
  qwenpi_v3) VLA_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_pi05/bin/python ;;
  qwengr00t) VLA_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_gr00t/bin/python ;;
  *) exit 2 ;;
esac
case "${CONDITION}" in
  profile-latency)
    BOOTSTRAP="${DATA}/profile-latency/${MODEL}"
    EVAL_CONFIG="${SOURCE}/evals/final-profile/eval_config.yaml"
    LATENCY_CONFIG="${DATA}/configs/profile-${MODEL}.yaml"
    ;;
  fixed-[1234])
    BOOTSTRAP="${DATA}/${CONDITION}"
    EVAL_CONFIG="${SOURCE}/evals/final-${CONDITION}/eval_config.yaml"
    LATENCY_CONFIG="${DATA}/configs/${CONDITION}.yaml"
    ;;
  *) exit 2 ;;
esac

cd "${ROOT}"
set -a
. ./.env
set +a
export WANDB_ENTITY=saberrr-zju
export PYTHONPATH="${ROOT}:${ROOT}/third_party/starVLA"
UV_EXEC="${UV:-/home/ubuntu/.local/bin/uv}"
EXTRA=()
if [[ "${MODE}" == smoke ]]; then EXTRA+=(--smoke); fi

exec "${VLA_PY}" -m scripts.tasks.mikasa_intercept_grab_fast.run_dagger \
  --model "${MODEL}" --condition "${CONDITION}" \
  --source-run "${SOURCE}" --bootstrap "${BOOTSTRAP}" \
  --eval-config "${EVAL_CONFIG}" --latency-config "${LATENCY_CONFIG}" \
  --teacher-checkpoint "${TEACHER}" --output "${OUTPUT}" \
  --vla-python "${VLA_PY}" --task-python "${TASK_PY}" --uv "${UV_EXEC}" \
  --train-devices "${TRAIN_DEVICES}" --inference-gpu "${INFERENCE_GPU}" \
  --simulator-gpu "${SIMULATOR_GPU}" \
  --gpu-lock "/tmp/mikasa-h8-dagger-gpu${SIMULATOR_GPU}.lock" \
  --port "${PORT}" "${EXTRA[@]}"
