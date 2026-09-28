#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SECOND_HOST="ubuntu@34.131.91.104"
SECOND_REPO="/home/ubuntu/codex/mikasa-h8"
DATA_ROOT="${REPO_ROOT}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v2"
OUTPUT_ROOT="${REPO_ROOT}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned"
PROFILE_SCRIPT="scripts/tasks/mikasa_intercept_grab_fast/run_profile_v2_h8.sh"
FIXED_SCRIPT="scripts/tasks/mikasa_intercept_grab_fast/run_fixed_v2_h8.sh"

export QWENOFT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/python
export QWENPI_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_pi05/bin/python
export QWENGROOT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_gr00t/bin/python
export MIKASA_PY="${REPO_ROOT}/third_party/MIKASA-Robo/.venv/bin/python"
export UV=/home/ubuntu/.local/bin/uv

cd "${REPO_ROOT}"
set -a
. "${REPO_ROOT}/.env"
set +a
export WANDB_ENTITY=saberrr-zju
export WANDB_MODE=online
export WANDB_RESUME=allow
mkdir -p "${OUTPUT_ROOT}"
rm -f "${OUTPUT_ROOT}/h8-v2-six-way.complete" "${OUTPUT_ROOT}/h8-v2-six-way.failed"

WANDB_RUN_ID=mikasa-h8-v2-profile-qwenoft-syncfix-20260922 \
PROFILE_MODELS=qwenoft PROFILE_TRAIN_DEVICES=0,1 PROFILE_TRAIN_PORTS=29520 \
  bash "${PROFILE_SCRIPT}" train-eval \
  > "${OUTPUT_ROOT}/six-way-first-profile-qwenoft.log" 2>&1 &
P1=$!

WANDB_RUN_ID=mikasa-h8-v2-profile-qwenpi-v3-syncfix-20260922 \
PROFILE_MODELS=qwenpi_v3 PROFILE_TRAIN_DEVICES=2,3 PROFILE_TRAIN_PORTS=29522 \
  bash "${PROFILE_SCRIPT}" train-eval \
  > "${OUTPUT_ROOT}/six-way-first-profile-qwenpi-v3.log" 2>&1 &
P2=$!

WANDB_RUN_ID=mikasa-h8-v2-fixed-qwenoft-syncfix-20260922 \
FIXED_MODELS=qwenoft FIXED_TRAIN_DEVICES=4,5 FIXED_TRAIN_PORTS=29530 \
  bash "${FIXED_SCRIPT}" train-eval \
  > "${OUTPUT_ROOT}/six-way-first-fixed-qwenoft.log" 2>&1 &
P3=$!

WANDB_RUN_ID=mikasa-h8-v2-fixed-qwenpi-v3-syncfix-20260922 \
FIXED_MODELS=qwenpi_v3 FIXED_TRAIN_DEVICES=6,7 FIXED_TRAIN_PORTS=29532 \
  bash "${FIXED_SCRIPT}" train-eval \
  > "${OUTPUT_ROOT}/six-way-first-fixed-qwenpi-v3.log" 2>&1 &
P4=$!

ssh "${SECOND_HOST}" "mkdir -p '${SECOND_REPO}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned'"

ssh "${SECOND_HOST}" "cd '${SECOND_REPO}' && set -a && . ./.env && set +a && \
  export QWENOFT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/python \
  QWENPI_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_pi05/bin/python \
  QWENGROOT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_gr00t/bin/python \
  MIKASA_PY='${SECOND_REPO}/third_party/MIKASA-Robo/.venv/bin/python' \
  UV=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/uv \
  WANDB_ENTITY=saberrr-zju WANDB_MODE=online WANDB_RESUME=allow \
  WANDB_RUN_ID=mikasa-h8-v2-profile-qwengr00t-syncfix-20260922 PROFILE_MODELS=qwengr00t \
  PROFILE_TRAIN_DEVICES=4,5 PROFILE_TRAIN_PORTS=29524 && \
  bash '${PROFILE_SCRIPT}' train-eval" \
  > "${OUTPUT_ROOT}/six-way-second-profile-qwengr00t.log" 2>&1 &
P5=$!

ssh "${SECOND_HOST}" "cd '${SECOND_REPO}' && set -a && . ./.env && set +a && \
  export QWENOFT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/python \
  QWENPI_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_pi05/bin/python \
  QWENGROOT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_gr00t/bin/python \
  MIKASA_PY='${SECOND_REPO}/third_party/MIKASA-Robo/.venv/bin/python' \
  UV=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/uv \
  WANDB_ENTITY=saberrr-zju WANDB_MODE=online WANDB_RESUME=allow \
  WANDB_RUN_ID=mikasa-h8-v2-fixed-qwengr00t-syncfix-20260922 FIXED_MODELS=qwengr00t \
  FIXED_TRAIN_DEVICES=6,7 FIXED_TRAIN_PORTS=29534 && \
  bash '${FIXED_SCRIPT}' train-eval" \
  > "${OUTPUT_ROOT}/six-way-second-fixed-qwengr00t.log" 2>&1 &
P6=$!

failed=0
for pid in "${P1}" "${P2}" "${P3}" "${P4}" "${P5}" "${P6}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  touch "${OUTPUT_ROOT}/h8-v2-six-way.failed"
  exit 1
fi

rm -rf \
  "${OUTPUT_ROOT}/profile-latency/qwengr00t" \
  "${OUTPUT_ROOT}/fixed-2/qwengr00t"
rsync -a \
  "${SECOND_HOST}:${SECOND_REPO}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned/profile-latency/qwengr00t" \
  "${OUTPUT_ROOT}/profile-latency/"
rsync -a \
  "${SECOND_HOST}:${SECOND_REPO}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned/fixed-2/qwengr00t" \
  "${OUTPUT_ROOT}/fixed-2/"

bash "${PROFILE_SCRIPT}" verify > "${OUTPUT_ROOT}/profile-v2-verification.json"
bash "${FIXED_SCRIPT}" verify > "${OUTPUT_ROOT}/fixed-v2-verification.json"
bash "${PROFILE_SCRIPT}" publish > "${OUTPUT_ROOT}/profile-v2-publish.log" 2>&1
touch "${OUTPUT_ROOT}/h8-v2-six-way.complete"
