#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <accepted-fasttd3-checkpoint> <latency-config>" >&2
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
MAX_STEPS="${BALANCE_VLA_MAX_STEPS:-15000}"
STEP_INTERVAL="${BALANCE_VLA_STEP_INTERVAL:-500}"
START_STEP="${BALANCE_VLA_START_STEP:-500}"
NUM_GPUS="${BALANCE_VLA_NUM_GPUS:-2}"
GLOBAL_BATCH_SIZE="${BALANCE_VLA_GLOBAL_BATCH_SIZE:-128}"
MAX_EPISODES="${BALANCE_EXPORT_MAX_EPISODES:-100}"
VAL_FRACTION="${BALANCE_EXPORT_VAL_FRACTION:-0.1}"
NUM_ENVS="${BALANCE_EXPORT_NUM_ENVS:-16}"
REPLAY_NUM_ENVS="${BALANCE_EXPORT_REPLAY_NUM_ENVS:-16}"
IMAGE_WRITER_WORKERS="${BALANCE_EXPORT_IMAGE_WRITER_WORKERS:-16}"
ROW_BUFFER_SIZE="${BALANCE_EXPORT_ROW_BUFFER_SIZE:-8192}"
EXPORT_DEVICE="${BALANCE_EXPORT_DEVICE:-cuda}"
BASELINE_ARGS=()
if [[ -n "${BALANCE_BASELINE_MODEL_PATH:-}" ]]; then
  BASELINE_ARGS+=(--baseline-model-path "${BALANCE_BASELINE_MODEL_PATH}")
fi
LATENCY_CONFIG="$2"

cd "${REPO_ROOT}"
LATENCY_PROFILE="$("${GROOT_PY}" -c 'import sys; from latency_bench.core.config import load_config; print(load_config(sys.argv[1])["latency"]["profile_path"])' "${LATENCY_CONFIG}")"
"${GROOT_BIN}/hf" download nvidia/GR00T-N1.7-3B \
  --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --local-dir "${BASE_MODEL_PATH}"
uv run --project . python -m latency_bench.envs.humanoidbench_sonic_assets \
  --cache-dir "${ASSET_CACHE_DIR}"
MUJOCO_GL=egl uv run --project . python scripts/teachers/export_g1_sonic_gr00t.py \
  --output-root "${DATASET_ROOT}" \
  --teacher "${TEACHER_NAME}" \
  --checkpoint "$1" \
  --max-episodes "${MAX_EPISODES}" \
  --val-fraction "${VAL_FRACTION}" \
  --split-seed 0 \
  --num-envs "${NUM_ENVS}" \
  --replay-num-envs "${REPLAY_NUM_ENVS}" \
  --device "${EXPORT_DEVICE}" \
  --deterministic \
  --image-writer-workers "${IMAGE_WRITER_WORKERS}" \
  --row-buffer-size "${ROW_BUFFER_SIZE}" \
  --balance-teacher-observation-state \
  --latency-config "${LATENCY_CONFIG}" \
  --force

"${GROOT_PY}" scripts/official_gr00t/train_balance_simple_periodic_eval.py \
  --base-model-path "${BASE_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  "${BASELINE_ARGS[@]}" \
  --latency-config "${LATENCY_CONFIG}" \
  --latency-profile-path "${LATENCY_PROFILE}" \
  --num-gpus "${NUM_GPUS}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --start-step "${START_STEP}" \
  --max-step "${MAX_STEPS}" \
  --step-interval "${STEP_INTERVAL}"

printf 'MODEL_PATH=%s\n' "${OUTPUT_DIR}/${RUN_ID}/checkpoint-${MAX_STEPS}"
