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
export WANDB_ENTITY="${MIKASA_WANDB_ENTITY:-saberrr-zju}"
mkdir -p "${OUTPUT_ROOT}"

bootstrap_second_host() {
  ssh "${SECOND_HOST}" "mkdir -p '${SECOND_REPO}/third_party/MIKASA-Robo' '${SECOND_REPO}/third_party/starVLA'"
  rsync -aR --exclude='__pycache__' --exclude='*.pyc' \
    ./.env ./pyproject.toml \
    ./latency_bench ./scripts ./configs ./training \
    ./results/profiling/downloads/profiles \
    "${SECOND_HOST}:${SECOND_REPO}/"
  rsync -a --exclude='.git' --exclude='playground' --exclude='__pycache__' \
    third_party/starVLA/ "${SECOND_HOST}:${SECOND_REPO}/third_party/starVLA/"
  rsync -a --exclude='.git' --exclude='docs' --exclude='utils' --exclude='__pycache__' \
    third_party/MIKASA-Robo/ "${SECOND_HOST}:${SECOND_REPO}/third_party/MIKASA-Robo/"
}

bootstrap_second_host > "${OUTPUT_ROOT}/cross-host-bootstrap.log" 2>&1 &
BOOTSTRAP_PID=$!

PROFILE_MODELS='qwenoft;qwenpi_v3;qwengr00t' \
PROFILE_COLLECT_GPUS='0;1;2' \
  bash "${PROFILE_SCRIPT}" collect \
  > "${OUTPUT_ROOT}/profile-v2-collect-supervisor.log" 2>&1 &
PROFILE_COLLECT_PID=$!

FIXED_COLLECT_GPU=3 bash "${FIXED_SCRIPT}" collect \
  > "${OUTPUT_ROOT}/fixed-v2-collect-supervisor.log" 2>&1 &
FIXED_COLLECT_PID=$!

wait "${BOOTSTRAP_PID}"
wait "${PROFILE_COLLECT_PID}"
wait "${FIXED_COLLECT_PID}"

ssh "${SECOND_HOST}" \
  "rm -rf '${SECOND_REPO}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v2/profile-latency/qwengr00t' '${SECOND_REPO}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v2/fixed-2'"
ssh "${SECOND_HOST}" \
  "mkdir -p '${SECOND_REPO}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v2/profile-latency'"
rsync -a "${DATA_ROOT}/profile-latency/qwengr00t" \
  "${SECOND_HOST}:${SECOND_REPO}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v2/profile-latency/"
rsync -a "${DATA_ROOT}/fixed-2" \
  "${SECOND_HOST}:${SECOND_REPO}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v2/"

(
  PROFILE_MODELS=qwenoft PROFILE_TRAIN_DEVICES=0,1 PROFILE_TRAIN_PORTS=29520 \
    bash "${PROFILE_SCRIPT}" train-eval
  FIXED_MODELS=qwenoft FIXED_TRAIN_DEVICES=0,1 FIXED_TRAIN_PORTS=29530 \
    bash "${FIXED_SCRIPT}" train-eval
) > "${OUTPUT_ROOT}/first-h200-pair-01.log" 2>&1 &
PAIR_01_PID=$!

(
  PROFILE_MODELS=qwenpi_v3 PROFILE_TRAIN_DEVICES=2,3 PROFILE_TRAIN_PORTS=29522 \
    bash "${PROFILE_SCRIPT}" train-eval
  FIXED_MODELS=qwenpi_v3 FIXED_TRAIN_DEVICES=2,3 FIXED_TRAIN_PORTS=29532 \
    bash "${FIXED_SCRIPT}" train-eval
) > "${OUTPUT_ROOT}/first-h200-pair-23.log" 2>&1 &
PAIR_23_PID=$!

ssh "${SECOND_HOST}" "cd '${SECOND_REPO}' && set -a && . ./.env && set +a && \
  export QWENOFT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/python \
  QWENPI_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_pi05/bin/python \
  QWENGROOT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_gr00t/bin/python \
  MIKASA_PY='${SECOND_REPO}/third_party/MIKASA-Robo/.venv/bin/python' \
  UV=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/uv \
  WANDB_ENTITY='${WANDB_ENTITY}' PROFILE_MODELS=qwengr00t \
  PROFILE_TRAIN_DEVICES=4,5 PROFILE_TRAIN_PORTS=29524 && \
  bash '${PROFILE_SCRIPT}' train-eval" \
  > "${OUTPUT_ROOT}/second-h200-pair-45.log" 2>&1 &
PAIR_45_PID=$!

ssh "${SECOND_HOST}" "cd '${SECOND_REPO}' && set -a && . ./.env && set +a && \
  export QWENOFT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/python \
  QWENPI_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_pi05/bin/python \
  QWENGROOT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_gr00t/bin/python \
  MIKASA_PY='${SECOND_REPO}/third_party/MIKASA-Robo/.venv/bin/python' \
  UV=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/uv \
  WANDB_ENTITY='${WANDB_ENTITY}' FIXED_MODELS=qwengr00t \
  FIXED_TRAIN_DEVICES=6,7 FIXED_TRAIN_PORTS=29534 && \
  bash '${FIXED_SCRIPT}' train-eval" \
  > "${OUTPUT_ROOT}/second-h200-pair-67.log" 2>&1 &
PAIR_67_PID=$!

failed=0
for pid in "${PAIR_01_PID}" "${PAIR_23_PID}" "${PAIR_45_PID}" "${PAIR_67_PID}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
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
touch "${OUTPUT_ROOT}/h8-v2-cross-host.complete"
