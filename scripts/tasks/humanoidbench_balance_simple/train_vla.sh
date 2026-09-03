#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <accepted-fasttd3-checkpoint>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TASK_ROOT="${BALANCE_TASK_ROOT:-${REPO_ROOT}/outputs/tasks/humanoidbench_balance_simple}"
ASSET_CACHE_DIR="${BALANCE_ASSET_CACHE_DIR:-${REPO_ROOT}/results/humanoidbench_g1_sonic_assets}"
BASE_MODEL_PATH="${BALANCE_BASE_MODEL_PATH:-${REPO_ROOT}/outputs/official_gr00t/models/GR00T-N1.7-3B}"
GROOT_PY="${BALANCE_GROOT_PY:-${REPO_ROOT}/third_party/Isaac-GR00T/.venv/bin/python}"
GROOT_BIN="$(dirname "${GROOT_PY}")"
DATASET_ROOT="${TASK_ROOT}/teacher_dataset"
TEACHER_NAME="g1_balance_simple_raw_ramp_fasttd3_25m_43d_teacher_obs198"
DATASET_PATH="${DATASET_ROOT}/${TEACHER_NAME}/lerobot"
OUTPUT_DIR="${BALANCE_VLA_OUTPUT_DIR:-${TASK_ROOT}/vla_obs198_h40}"
RUN_ID="${BALANCE_RUN_ID:-g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_15k}"

cd "${REPO_ROOT}"
"${GROOT_BIN}/hf" download nvidia/GR00T-N1.7-3B \
  --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --local-dir "${BASE_MODEL_PATH}"
uv run --project . python -m latency_bench.envs.humanoidbench_sonic_assets \
  --cache-dir "${ASSET_CACHE_DIR}"
MUJOCO_GL=egl uv run --project . python scripts/teachers/export_g1_sonic_gr00t.py \
  --output-root "${DATASET_ROOT}" \
  --teacher "${TEACHER_NAME}" \
  --checkpoint "$1" \
  --max-episodes 100 \
  --val-fraction 0.1 \
  --split-seed 0 \
  --num-envs 16 \
  --replay-num-envs 16 \
  --device cuda \
  --deterministic \
  --image-writer-workers 16 \
  --row-buffer-size 8192 \
  --balance-teacher-observation-state \
  --force

"${GROOT_PY}" scripts/official_gr00t/train_balance_simple_periodic_eval.py \
  --base-model-path "${BASE_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --start-step "${BALANCE_START_STEP:-500}"
