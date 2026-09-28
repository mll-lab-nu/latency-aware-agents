#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${MIKASA_H8_OUTPUT_ROOT:-${REPO_ROOT}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-zero-latency}"
QWENOFT_PY_EXEC="$(command -v "${QWENOFT_PY:?Set QWENOFT_PY}")"
QWENPI_PY_EXEC="$(command -v "${QWENPI_PY:?Set QWENPI_PY}")"
QWENGROOT_PY_EXEC="$(command -v "${QWENGROOT_PY:?Set QWENGROOT_PY}")"
MIKASA_PY_EXEC="${MIKASA_PY:-${REPO_ROOT}/third_party/MIKASA-Robo/.venv/bin/python}"
RUN_ID="h8-zero-latency-seed0"
DATASET_ROOT="${REPO_ROOT}/outputs/starvla/datasets/mikasa_intercept_grab_fast/teacher-rollouts-h8-state-v1"
SMOKE_ROOT="${OUTPUT_ROOT}/smoke"
SMOKE_ID="h8-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

"${QWENOFT_PY_EXEC}" -m pytest \
  tests/scripts/test_mikasa_h8.py \
  tests/data/test_starvla_tasks.py \
  tests/executors/test_simulated_executor.py \
  tests/scripts/test_starvla_train_task.py

if [[ ! -f "${DATASET_ROOT}/lerobot/task_contract.json" ]]; then
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.prepare_h8_dataset \
    --output-train "${DATASET_ROOT}/lerobot"
fi

CUDA_VISIBLE_DEVICES=1 VLA_PY="${QWENOFT_PY_EXEC}" \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_h8.sh qwenoft \
  --smoke --run-id "${SMOKE_ID}" --output-dir "${SMOKE_ROOT}/${SMOKE_ID}/qwenoft" \
  --dataset-dir "${DATASET_ROOT}" --num-gpus 1 --micro-batch 128 --port 29421 \
  --gradient-checkpointing --inference-device cuda:1 --simulator-device cuda:0 \
  --eval-cuda-visible-devices 0,1 \
  --task-python "${MIKASA_PY_EXEC}" &
QWENOFT_SMOKE_PID=$!

CUDA_VISIBLE_DEVICES=2 VLA_PY="${QWENPI_PY_EXEC}" \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_h8.sh qwenpi_v3 \
  --smoke --run-id "${SMOKE_ID}" --output-dir "${SMOKE_ROOT}/${SMOKE_ID}/qwenpi_v3" \
  --dataset-dir "${DATASET_ROOT}" --num-gpus 1 --micro-batch 128 --port 29423 \
  --gradient-checkpointing --inference-device cuda:1 --simulator-device cuda:0 \
  --eval-cuda-visible-devices 0,2 \
  --task-python "${MIKASA_PY_EXEC}" &
QWENPI_SMOKE_PID=$!

CUDA_VISIBLE_DEVICES=3 VLA_PY="${QWENGROOT_PY_EXEC}" \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_h8.sh qwengr00t \
  --smoke --run-id "${SMOKE_ID}" --output-dir "${SMOKE_ROOT}/${SMOKE_ID}/qwengr00t" \
  --dataset-dir "${DATASET_ROOT}" --num-gpus 1 --micro-batch 64 --port 29425 \
  --gradient-checkpointing --inference-device cuda:1 --simulator-device cuda:0 \
  --eval-cuda-visible-devices 0,3 \
  --task-python "${MIKASA_PY_EXEC}" &
QWENGROOT_SMOKE_PID=$!

wait "${QWENOFT_SMOKE_PID}"
wait "${QWENPI_SMOKE_PID}"
wait "${QWENGROOT_SMOKE_PID}"

CUDA_VISIBLE_DEVICES=1 VLA_PY="${QWENOFT_PY_EXEC}" \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_h8.sh qwenoft \
  --run-id "${RUN_ID}" --output-dir "${OUTPUT_ROOT}/qwenoft" \
  --dataset-dir "${DATASET_ROOT}" --num-gpus 1 --micro-batch 128 --port 29521 --gradient-checkpointing \
  --inference-device cuda:1 --simulator-device cuda:0 --eval-cuda-visible-devices 0,1 \
  --task-python "${MIKASA_PY_EXEC}" &
QWENOFT_PID=$!

CUDA_VISIBLE_DEVICES=2 VLA_PY="${QWENPI_PY_EXEC}" \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_h8.sh qwenpi_v3 \
  --run-id "${RUN_ID}" --output-dir "${OUTPUT_ROOT}/qwenpi_v3" \
  --dataset-dir "${DATASET_ROOT}" --num-gpus 1 --micro-batch 128 --port 29523 --gradient-checkpointing \
  --inference-device cuda:1 --simulator-device cuda:0 --eval-cuda-visible-devices 0,2 \
  --task-python "${MIKASA_PY_EXEC}" &
QWENPI_PID=$!

CUDA_VISIBLE_DEVICES=3 VLA_PY="${QWENGROOT_PY_EXEC}" \
  bash scripts/tasks/mikasa_intercept_grab_fast/train_h8.sh qwengr00t \
  --run-id "${RUN_ID}" --output-dir "${OUTPUT_ROOT}/qwengr00t" \
  --dataset-dir "${DATASET_ROOT}" --num-gpus 1 --micro-batch 64 --port 29525 --gradient-checkpointing \
  --inference-device cuda:1 --simulator-device cuda:0 --eval-cuda-visible-devices 0,3 \
  --task-python "${MIKASA_PY_EXEC}" &
QWENGROOT_PID=$!

wait "${QWENOFT_PID}"
wait "${QWENPI_PID}"
wait "${QWENGROOT_PID}"
