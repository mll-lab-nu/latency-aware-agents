#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned"
DATA_ROOT="${REPO_ROOT}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v2"
PROFILE_COLLECT_PID_FILE="${OUTPUT_ROOT}/profile-v2-collect-supervisor.pid"
FIXED_COLLECT_PID_FILE="${OUTPUT_ROOT}/fixed-v2-collect-supervisor.pid"
QWENOFT_PY_EXEC="${QWENOFT_PY:?Set QWENOFT_PY}"

cd "${REPO_ROOT}"
set -a
. "${REPO_ROOT}/.env"
set +a
export WANDB_ENTITY="${MIKASA_WANDB_ENTITY:-saberrr-zju}"

profile_collect_pid="$(<"${PROFILE_COLLECT_PID_FILE}")"
while kill -0 "${profile_collect_pid}" 2>/dev/null; do
  sleep 60
done
fixed_collect_pid="$(<"${FIXED_COLLECT_PID_FILE}")"
while kill -0 "${fixed_collect_pid}" 2>/dev/null; do
  sleep 60
done
for model in qwenoft qwenpi_v3 qwengr00t; do
  test -f "${DATA_ROOT}/profile-latency/${model}/dataset_audit.json"
done
test -f "${DATA_ROOT}/fixed-2/dataset_audit.json"

bash scripts/tasks/mikasa_intercept_grab_fast/run_profile_v2_h8.sh train-eval
bash scripts/tasks/mikasa_intercept_grab_fast/run_profile_v2_h8.sh verify
bash scripts/tasks/mikasa_intercept_grab_fast/run_profile_v2_h8.sh publish
FIXED_MODELS='qwenoft;qwenpi_v3;qwengr00t' \
FIXED_TRAIN_DEVICES='0,1;2,3;4,5' FIXED_TRAIN_PORTS='29530;29532;29534' \
  bash scripts/tasks/mikasa_intercept_grab_fast/run_fixed_v2_h8.sh train-eval \
  > "${OUTPUT_ROOT}/fixed-v2-supervisor.log" 2>&1
"${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.verify_fixed_v2_h8 \
  --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --condition fixed-2 --models qwenoft qwenpi_v3 qwengr00t \
  > "${OUTPUT_ROOT}/fixed-v2-verification.json"
touch "${OUTPUT_ROOT}/h8-v2-queue.complete"
