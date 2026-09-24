#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:?Usage: run_fixed_v2_h8.sh collect|train-eval|verify|all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${MIKASA_H8_DATA_ROOT:-${REPO_ROOT}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v2}"
OUTPUT_ROOT="${MIKASA_H8_OUTPUT_ROOT:-${REPO_ROOT}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned}"
TEACHER="${REPO_ROOT}/outputs/starvla/artifacts/models/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt"
PACKAGER_PROJECT="${REPO_ROOT}/third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot"
CONDITION="${FIXED_CONDITION:-fixed-2}"
RUN_ID="${MIKASA_H8_FIXED_RUN_ID:-h8-conditioned-seed0-${CONDITION}}"
QWENOFT_PY_EXEC="${QWENOFT_PY:?Set QWENOFT_PY}"
MIKASA_PY_EXEC="${MIKASA_PY:?Set MIKASA_PY}"
UV_EXEC="${UV:?Set UV}"
FIXED_COLLECT_GPU="${FIXED_COLLECT_GPU:-0}"
IFS=';' read -r -a FIXED_MODELS_ARRAY <<< "${FIXED_MODELS:-qwenoft}"
IFS=';' read -r -a FIXED_TRAIN_DEVICES_ARRAY <<< "${FIXED_TRAIN_DEVICES:-0,1}"
IFS=';' read -r -a FIXED_TRAIN_PORTS_ARRAY <<< "${FIXED_TRAIN_PORTS:-29530}"

cd "${REPO_ROOT}"

collect() {
  mkdir -p "${DATA_ROOT}/configs"
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.write_conditioned_latency_config \
    --condition "${CONDITION}" --output "${DATA_ROOT}/configs/${CONDITION}.yaml"
  CUDA_VISIBLE_DEVICES="${FIXED_COLLECT_GPU}" PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/MIKASA-Robo" \
    "${MIKASA_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 \
    --condition "${CONDITION}" --model shared \
    --latency-config "${DATA_ROOT}/configs/${CONDITION}.yaml" \
    --checkpoint "${TEACHER}" --output "${DATA_ROOT}/${CONDITION}/raw" \
    --train-targets 64288 --validation-targets 6884 --num-envs 128 \
    > "${DATA_ROOT}/${CONDITION}-collect.log" 2>&1
  PYTHONPATH="${REPO_ROOT}" "${UV_EXEC}" run --frozen --project "${PACKAGER_PROJECT}" python \
    -m scripts.tasks.mikasa_intercept_grab_fast.prepare_conditioned_h8_dataset \
    --raw-root "${DATA_ROOT}/${CONDITION}/raw" --output-train "${DATA_ROOT}/${CONDITION}/lerobot"
  PYTHONPATH="${REPO_ROOT}" "${UV_EXEC}" run --frozen --project "${PACKAGER_PROJECT}" python \
    -m scripts.tasks.mikasa_intercept_grab_fast.audit_conditioned_h8_dataset \
    --condition "${CONDITION}" --raw-root "${DATA_ROOT}/${CONDITION}/raw" \
    --dataset "${DATA_ROOT}/${CONDITION}/lerobot"
}

train_eval_model() {
  local model="$1"
  local vla_python="$2"
  local devices="$3"
  local port="$4"
  local output="${OUTPUT_ROOT}/${CONDITION}/${model}"
  local checkpointing_flag="--no-gradient-checkpointing"
  if [[ "${model}" != qwenoft ]]; then
    checkpointing_flag="--gradient-checkpointing"
  fi
  mkdir -p "${OUTPUT_ROOT}"
  CUDA_VISIBLE_DEVICES="${devices}" VLA_PY="${vla_python}" \
    bash scripts/tasks/mikasa_intercept_grab_fast/train_conditioned_h8.sh "${model}" \
    --condition "${CONDITION}" --train-only --no-oom-retry "${checkpointing_flag}" \
    --run-id "${RUN_ID}" --output-dir "${output}" \
    --dataset-dir "${DATA_ROOT}/${CONDITION}" \
    --num-gpus 2 --micro-batch 64 --port "${port}" --task-python "${MIKASA_PY_EXEC}" \
    > "${OUTPUT_ROOT}/${CONDITION}-${model}-v2.log" 2>&1
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.evaluate_h8_checkpoint \
    --model "${model}" --run-dir "${output}" --vla-python "${vla_python}" \
    --task-python "${MIKASA_PY_EXEC}" --eval-cuda-visible-devices "${devices}" \
    --setting "${CONDITION}" --inference-device cuda:1 --simulator-device cuda:0 \
    --seed 4242424242 --episodes 200 --label "final-${CONDITION}" --skip-wandb
}

model_python() {
  case "$1" in
    qwenoft) echo "${QWENOFT_PY_EXEC}" ;;
    qwenpi_v3) echo "${QWENPI_PY:?Set QWENPI_PY}" ;;
    qwengr00t) echo "${QWENGROOT_PY:?Set QWENGROOT_PY}" ;;
  esac
}

train_eval() {
  local pids=()
  local index model
  for index in "${!FIXED_MODELS_ARRAY[@]}"; do
    model="${FIXED_MODELS_ARRAY[${index}]}"
    train_eval_model \
      "${model}" "$(model_python "${model}")" \
      "${FIXED_TRAIN_DEVICES_ARRAY[${index}]}" \
      "${FIXED_TRAIN_PORTS_ARRAY[${index}]}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  return "${failed}"
}

verify() {
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.verify_fixed_v2_h8 \
    --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" \
    --condition "${CONDITION}" --models "${FIXED_MODELS_ARRAY[@]}"
}

case "${PHASE}" in
  collect)
    collect
    ;;
  train-eval)
    train_eval
    ;;
  verify) verify ;;
  all)
    collect
    train_eval
    verify
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    exit 2
    ;;
esac
