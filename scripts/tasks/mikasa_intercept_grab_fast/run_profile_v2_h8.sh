#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:?Usage: run_profile_v2_h8.sh smoke|collect|train-eval|verify|publish}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${MIKASA_H8_DATA_ROOT:-${REPO_ROOT}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v2}"
SMOKE_DATA_ROOT="${REPO_ROOT}/outputs/starvla/smoke/mikasa_intercept_grab_fast/conditioned-h8-v2-data"
SMOKE_OUTPUT_ROOT="${REPO_ROOT}/outputs/starvla/smoke/mikasa_intercept_grab_fast/profile-v2"
OUTPUT_ROOT="${MIKASA_H8_OUTPUT_ROOT:-${REPO_ROOT}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned}"
TEACHER="${REPO_ROOT}/outputs/starvla/artifacts/models/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt"
PACKAGER_PROJECT="${REPO_ROOT}/third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot"
RUN_ID="${MIKASA_H8_PROFILE_RUN_ID:-h8-conditioned-seed0-profile}"
QWENOFT_PY_EXEC="${QWENOFT_PY:?Set QWENOFT_PY}"
QWENPI_PY_EXEC="${QWENPI_PY:?Set QWENPI_PY}"
QWENGROOT_PY_EXEC="${QWENGROOT_PY:?Set QWENGROOT_PY}"
MIKASA_PY_EXEC="${MIKASA_PY:?Set MIKASA_PY}"
UV_EXEC="${UV:?Set UV}"
IFS=';' read -r -a PROFILE_MODELS_ARRAY <<< "${PROFILE_MODELS:-qwenoft;qwenpi_v3;qwengr00t}"
IFS=';' read -r -a PROFILE_COLLECT_GPUS_ARRAY <<< "${PROFILE_COLLECT_GPUS:-0;2;4}"
IFS=';' read -r -a PROFILE_TRAIN_DEVICES_ARRAY <<< "${PROFILE_TRAIN_DEVICES:-0,1;2,3;4,5}"
IFS=';' read -r -a PROFILE_TRAIN_PORTS_ARRAY <<< "${PROFILE_TRAIN_PORTS:-29520;29522;29524}"

cd "${REPO_ROOT}"

write_configs() {
  local root="$1"
  local model
  mkdir -p "${root}/configs"
  for model in "${PROFILE_MODELS_ARRAY[@]}"; do
    "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.write_conditioned_latency_config \
      --condition profile-latency --model "${model}" \
      --output "${root}/configs/profile-${model}.yaml"
  done
}

collect_package_model() {
  local root="$1"
  local train_targets="$2"
  local validation_targets="$3"
  local model="$4"
  local gpu="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/MIKASA-Robo" \
    "${MIKASA_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 \
    --condition profile-latency --model "${model}" \
    --latency-config "${root}/configs/profile-${model}.yaml" \
    --checkpoint "${TEACHER}" --output "${root}/profile-latency/${model}/raw" \
    --train-targets "${train_targets}" --validation-targets "${validation_targets}" \
    --num-envs 128 \
    > "${root}/profile-${model}-collect.log" 2>&1
  PYTHONPATH="${REPO_ROOT}" "${UV_EXEC}" run --frozen --project "${PACKAGER_PROJECT}" python \
    -m scripts.tasks.mikasa_intercept_grab_fast.prepare_conditioned_h8_dataset \
    --raw-root "${root}/profile-latency/${model}/raw" \
    --output-train "${root}/profile-latency/${model}/lerobot" \
    > "${root}/profile-${model}-package.log" 2>&1
  PYTHONPATH="${REPO_ROOT}" "${UV_EXEC}" run --frozen --project "${PACKAGER_PROJECT}" python \
    -m scripts.tasks.mikasa_intercept_grab_fast.audit_conditioned_h8_dataset \
    --condition profile-latency --raw-root "${root}/profile-latency/${model}/raw" \
    --dataset "${root}/profile-latency/${model}/lerobot" \
    >> "${root}/profile-${model}-package.log" 2>&1
}

collect_and_package() {
  local root="$1"
  local train_targets="$2"
  local validation_targets="$3"
  local failed=0
  local pids=()
  write_configs "${root}"
  for index in "${!PROFILE_MODELS_ARRAY[@]}"; do
    collect_package_model \
      "${root}" "${train_targets}" "${validation_targets}" \
      "${PROFILE_MODELS_ARRAY[${index}]}" "${PROFILE_COLLECT_GPUS_ARRAY[${index}]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  return "${failed}"
}

smoke_model() {
  local model="$1"
  local vla_python="$2"
  local devices="$3"
  local port="$4"
  local output="${SMOKE_OUTPUT_ROOT}/${model}"
  CUDA_VISIBLE_DEVICES="${devices}" VLA_PY="${vla_python}" \
    bash scripts/tasks/mikasa_intercept_grab_fast/train_conditioned_h8.sh "${model}" \
    --condition profile-latency --smoke --train-only --no-oom-retry --gradient-checkpointing \
    --run-id "smoke-profile-v2-${model}" --output-dir "${output}" \
    --dataset-dir "${SMOKE_DATA_ROOT}/profile-latency/${model}" \
    --num-gpus 2 --micro-batch 64 --port "${port}" \
    --task-python "${MIKASA_PY_EXEC}" \
    > "${SMOKE_OUTPUT_ROOT}/${model}.log" 2>&1
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.evaluate_h8_checkpoint \
    --model "${model}" --run-dir "${output}" --vla-python "${vla_python}" \
    --task-python "${MIKASA_PY_EXEC}" --eval-cuda-visible-devices "${devices}" \
    --setting profile --inference-device cuda:1 --simulator-device cuda:0 \
    --seed 4242424242 --episodes 2 --label final-profile --smoke \
    >> "${SMOKE_OUTPUT_ROOT}/${model}.log" 2>&1
}

train_eval_model() {
  local model="$1"
  local vla_python="$2"
  local devices="$3"
  local port="$4"
  local output="${OUTPUT_ROOT}/profile-latency/${model}"
  local checkpointing_flag="--no-gradient-checkpointing"
  if [[ "${model}" != qwenoft ]]; then
    checkpointing_flag="--gradient-checkpointing"
  fi
  mkdir -p "${OUTPUT_ROOT}"
  CUDA_VISIBLE_DEVICES="${devices}" VLA_PY="${vla_python}" \
    bash scripts/tasks/mikasa_intercept_grab_fast/train_conditioned_h8.sh "${model}" \
    --condition profile-latency --train-only --no-oom-retry "${checkpointing_flag}" \
    --run-id "${RUN_ID}" --output-dir "${output}" \
    --dataset-dir "${DATA_ROOT}/profile-latency/${model}" \
    --num-gpus 2 --micro-batch 64 --port "${port}" --task-python "${MIKASA_PY_EXEC}" \
    > "${OUTPUT_ROOT}/profile-${model}-v2.log" 2>&1
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.evaluate_h8_checkpoint \
    --model "${model}" --run-dir "${output}" --vla-python "${vla_python}" \
    --task-python "${MIKASA_PY_EXEC}" --eval-cuda-visible-devices "${devices}" \
    --setting profile --inference-device cuda:1 --simulator-device cuda:0 \
    --seed 4242424242 --episodes 200 --label final-profile --skip-wandb
}

model_python() {
  case "$1" in
    qwenoft) echo "${QWENOFT_PY_EXEC}" ;;
    qwenpi_v3) echo "${QWENPI_PY_EXEC}" ;;
    qwengr00t) echo "${QWENGROOT_PY_EXEC}" ;;
  esac
}

case "${PHASE}" in
  smoke)
    rm -rf "${SMOKE_DATA_ROOT}" "${SMOKE_OUTPUT_ROOT}"
    mkdir -p "${SMOKE_DATA_ROOT}" "${SMOKE_OUTPUT_ROOT}"
    collect_and_package "${SMOKE_DATA_ROOT}" 32 16
    smoke_model qwenoft "${QWENOFT_PY_EXEC}" 0,1 29620 & P1=$!
    smoke_model qwenpi_v3 "${QWENPI_PY_EXEC}" 2,3 29622 & P2=$!
    smoke_model qwengr00t "${QWENGROOT_PY_EXEC}" 4,5 29624 & P3=$!
    failed=0
    for pid in "${P1}" "${P2}" "${P3}"; do wait "${pid}" || failed=1; done
    exit "${failed}"
    ;;
  collect)
    collect_and_package "${DATA_ROOT}" 64288 6884
    ;;
  train-eval)
    pids=()
    for index in "${!PROFILE_MODELS_ARRAY[@]}"; do
      model="${PROFILE_MODELS_ARRAY[${index}]}"
      train_eval_model \
        "${model}" "$(model_python "${model}")" \
        "${PROFILE_TRAIN_DEVICES_ARRAY[${index}]}" \
        "${PROFILE_TRAIN_PORTS_ARRAY[${index}]}" &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    exit "${failed}"
    ;;
  verify)
    "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.verify_profile_v2_h8 \
      --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}"
    ;;
  publish)
    "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.verify_profile_v2_h8 \
      --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}"
    set -a
    . "${REPO_ROOT}/.env"
    set +a
    for model in qwenoft qwenpi_v3 qwengr00t; do
      "${QWENOFT_PY_EXEC}" -m scripts.starvla.train_task \
        --task mikasa_conditioned_h8 --model "${model}" --condition profile-latency \
        --run-id "${RUN_ID}" --output-dir "${OUTPUT_ROOT}/profile-latency/${model}" \
        --publish-only
    done
    "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.verify_profile_v2_h8 \
      --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}" --hub
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    exit 2
    ;;
esac
