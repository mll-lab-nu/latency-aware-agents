#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_ROOT="${MIKASA_CONDITIONED_OUTPUT_ROOT:-${REPO_ROOT}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned}"
ZERO_BASELINE_ROOT="${MIKASA_H8_ZERO_OUTPUT_ROOT:-${REPO_ROOT}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-zero-latency}"
DATA_ROOT="${MIKASA_CONDITIONED_DATA_ROOT:-${REPO_ROOT}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v1}"
QWENOFT_PY_EXEC="$(command -v "${QWENOFT_PY:-${VLA_PY:-python}}")"
QWENPI_PY_EXEC="$(command -v "${QWENPI_PY:-${VLA_PY:-python}}")"
QWENGROOT_PY_EXEC="$(command -v "${QWENGROOT_PY:-${VLA_PY:-python}}")"
MIKASA_PY_EXEC="${MIKASA_PY:-${REPO_ROOT}/third_party/MIKASA-Robo/.venv/bin/python}"
UV_EXEC="${UV:-${HOME}/.local/bin/uv}"
TEACHER="${MIKASA_ZERO_TEACHER:-${REPO_ROOT}/outputs/starvla/artifacts/models/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt}"
PACKAGER_PROJECT="${REPO_ROOT}/third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot"
RUN_ID="h8-conditioned-seed0"

mkdir -p "${DATA_ROOT}/configs" "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

collect() {
  for model in qwenoft qwenpi_v3 qwengr00t; do
    "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.write_conditioned_latency_config \
      --condition profile-latency --model "${model}" \
      --output "${DATA_ROOT}/configs/profile-${model}.yaml"
  done
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.write_conditioned_latency_config \
    --condition fixed-2 --output "${DATA_ROOT}/configs/fixed-2.yaml"

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/MIKASA-Robo" \
    "${MIKASA_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 \
    --condition profile-latency --model qwenoft --latency-config "${DATA_ROOT}/configs/profile-qwenoft.yaml" \
    --checkpoint "${TEACHER}" --output "${DATA_ROOT}/profile-latency/qwenoft/raw" \
    --num-envs 128 \
    > "${DATA_ROOT}/profile-qwenoft-collect.log" 2>&1 &
  P1=$!
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/MIKASA-Robo" \
    "${MIKASA_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 \
    --condition profile-latency --model qwenpi_v3 --latency-config "${DATA_ROOT}/configs/profile-qwenpi_v3.yaml" \
    --checkpoint "${TEACHER}" --output "${DATA_ROOT}/profile-latency/qwenpi_v3/raw" \
    --num-envs 128 \
    > "${DATA_ROOT}/profile-qwenpi_v3-collect.log" 2>&1 &
  P2=$!
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/MIKASA-Robo" \
    "${MIKASA_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 \
    --condition profile-latency --model qwengr00t --latency-config "${DATA_ROOT}/configs/profile-qwengr00t.yaml" \
    --checkpoint "${TEACHER}" --output "${DATA_ROOT}/profile-latency/qwengr00t/raw" \
    --num-envs 128 \
    > "${DATA_ROOT}/profile-qwengr00t-collect.log" 2>&1 &
  P3=$!
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/MIKASA-Robo" \
    "${MIKASA_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 \
    --condition fixed-2 --model shared --latency-config "${DATA_ROOT}/configs/fixed-2.yaml" \
    --checkpoint "${TEACHER}" --output "${DATA_ROOT}/fixed-2/raw" \
    --num-envs 128 \
    > "${DATA_ROOT}/fixed-2-collect.log" 2>&1 &
  P0=$!
  failed=0
  for pid in "${P0}" "${P1}" "${P2}" "${P3}"; do
    wait "${pid}" || failed=1
  done
  if [[ "${failed}" -ne 0 ]]; then
    return "${failed}"
  fi

  for model in qwenoft qwenpi_v3 qwengr00t; do
    PYTHONPATH="${REPO_ROOT}" "${UV_EXEC}" run --frozen --project "${PACKAGER_PROJECT}" python \
      -m scripts.tasks.mikasa_intercept_grab_fast.prepare_conditioned_h8_dataset \
      --raw-root "${DATA_ROOT}/profile-latency/${model}/raw" \
      --output-train "${DATA_ROOT}/profile-latency/${model}/lerobot"
    PYTHONPATH="${REPO_ROOT}" "${UV_EXEC}" run --frozen --project "${PACKAGER_PROJECT}" python \
      -m scripts.tasks.mikasa_intercept_grab_fast.audit_conditioned_h8_dataset \
      --condition profile-latency --raw-root "${DATA_ROOT}/profile-latency/${model}/raw" \
      --dataset "${DATA_ROOT}/profile-latency/${model}/lerobot"
  done
  PYTHONPATH="${REPO_ROOT}" "${UV_EXEC}" run --frozen --project "${PACKAGER_PROJECT}" python \
    -m scripts.tasks.mikasa_intercept_grab_fast.prepare_conditioned_h8_dataset \
    --raw-root "${DATA_ROOT}/fixed-2/raw" --output-train "${DATA_ROOT}/fixed-2/lerobot"
  PYTHONPATH="${REPO_ROOT}" "${UV_EXEC}" run --frozen --project "${PACKAGER_PROJECT}" python \
    -m scripts.tasks.mikasa_intercept_grab_fast.audit_conditioned_h8_dataset \
    --condition fixed-2 --raw-root "${DATA_ROOT}/fixed-2/raw" \
    --dataset "${DATA_ROOT}/fixed-2/lerobot"
}

train() {
  CUDA_VISIBLE_DEVICES=1 VLA_PY="${QWENOFT_PY_EXEC}" \
    bash scripts/tasks/mikasa_intercept_grab_fast/train_conditioned_h8.sh qwenoft \
    --condition profile-latency --train-only --no-oom-retry --run-id "${RUN_ID}-profile" \
    --output-dir "${OUTPUT_ROOT}/profile-latency/qwenoft" --dataset-dir "${DATA_ROOT}/profile-latency/qwenoft" \
    --num-gpus 1 --micro-batch 128 --port 29521 --gradient-checkpointing \
    --task-python "${MIKASA_PY_EXEC}" > "${OUTPUT_ROOT}/profile-qwenoft.log" 2>&1 &
  T1=$!
  CUDA_VISIBLE_DEVICES=2 VLA_PY="${QWENPI_PY_EXEC}" \
    bash scripts/tasks/mikasa_intercept_grab_fast/train_conditioned_h8.sh qwenpi_v3 \
    --condition profile-latency --train-only --no-oom-retry --run-id "${RUN_ID}-profile" \
    --output-dir "${OUTPUT_ROOT}/profile-latency/qwenpi_v3" --dataset-dir "${DATA_ROOT}/profile-latency/qwenpi_v3" \
    --num-gpus 1 --micro-batch 128 --port 29523 --gradient-checkpointing \
    --task-python "${MIKASA_PY_EXEC}" > "${OUTPUT_ROOT}/profile-qwenpi_v3.log" 2>&1 &
  T2=$!
  CUDA_VISIBLE_DEVICES=3 VLA_PY="${QWENGROOT_PY_EXEC}" \
    bash scripts/tasks/mikasa_intercept_grab_fast/train_conditioned_h8.sh qwengr00t \
    --condition profile-latency --train-only --no-oom-retry --run-id "${RUN_ID}-profile" \
    --output-dir "${OUTPUT_ROOT}/profile-latency/qwengr00t" --dataset-dir "${DATA_ROOT}/profile-latency/qwengr00t" \
    --num-gpus 1 --micro-batch 64 --port 29525 --gradient-checkpointing \
    --task-python "${MIKASA_PY_EXEC}" > "${OUTPUT_ROOT}/profile-qwengr00t.log" 2>&1 &
  T3=$!
  CUDA_VISIBLE_DEVICES=0 VLA_PY="${QWENOFT_PY_EXEC}" \
    bash scripts/tasks/mikasa_intercept_grab_fast/train_conditioned_h8.sh qwenoft \
    --condition fixed-2 --train-only --no-oom-retry --run-id "${RUN_ID}-fixed-2" \
    --output-dir "${OUTPUT_ROOT}/fixed-2/qwenoft" --dataset-dir "${DATA_ROOT}/fixed-2" \
    --num-gpus 1 --micro-batch 128 --port 29520 --gradient-checkpointing \
    --task-python "${MIKASA_PY_EXEC}" > "${OUTPUT_ROOT}/fixed-qwenoft.log" 2>&1 &
  T0=$!

  first_finished=""
  while [[ -z "${first_finished}" ]]; do
    for pid in "${T0}" "${T1}" "${T2}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        first_finished="${pid}"
        break
      fi
    done
    if [[ -z "${first_finished}" ]]; then
      sleep 10
    fi
  done
  if ! wait "${first_finished}"; then
    failed=1
    for pid in "${T0}" "${T1}" "${T2}" "${T3}"; do
      if [[ "${pid}" != "${first_finished}" ]]; then
        wait "${pid}" || failed=1
      fi
    done
    return "${failed}"
  fi

  while [[ -n "$(nvidia-smi --id=6 --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    sleep 60
  done

  CUDA_VISIBLE_DEVICES=6 VLA_PY="${QWENPI_PY_EXEC}" \
    bash scripts/tasks/mikasa_intercept_grab_fast/train_conditioned_h8.sh qwenpi_v3 \
    --condition fixed-2 --train-only --no-oom-retry --run-id "${RUN_ID}-fixed-2" \
    --output-dir "${OUTPUT_ROOT}/fixed-2/qwenpi_v3" --dataset-dir "${DATA_ROOT}/fixed-2" \
    --num-gpus 1 --micro-batch 128 --port 29526 --gradient-checkpointing \
    --task-python "${MIKASA_PY_EXEC}" > "${OUTPUT_ROOT}/fixed-qwenpi_v3.log" 2>&1 &
  T6=$!
  while [[ -n "$(nvidia-smi --id=7 --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    sleep 60
  done
  CUDA_VISIBLE_DEVICES=7 VLA_PY="${QWENGROOT_PY_EXEC}" \
    bash scripts/tasks/mikasa_intercept_grab_fast/train_conditioned_h8.sh qwengr00t \
    --condition fixed-2 --train-only --no-oom-retry --run-id "${RUN_ID}-fixed-2" \
    --output-dir "${OUTPUT_ROOT}/fixed-2/qwengr00t" --dataset-dir "${DATA_ROOT}/fixed-2" \
    --num-gpus 1 --micro-batch 64 --port 29527 --gradient-checkpointing \
    --task-python "${MIKASA_PY_EXEC}" > "${OUTPUT_ROOT}/fixed-qwengr00t.log" 2>&1 &
  T7=$!
  failed=0
  for pid in "${T0}" "${T1}" "${T2}" "${T3}" "${T6}" "${T7}"; do
    if [[ "${pid}" != "${first_finished}" ]]; then
      wait "${pid}" || failed=1
    fi
  done
  return "${failed}"
}

eval_model() {
  local model="$1"
  local vla_python="$2"
  local devices="$3"
  local inference_device="$4"
  local simulator_device="$5"
  local result_file
  local run_dir
  for condition_setting in "profile-latency profile" "fixed-2 fixed-2" "fixed-2 zero"; do
    read -r condition setting <<< "${condition_setting}"
    run_dir="${OUTPUT_ROOT}/${condition}/${model}"
    if [[ ! -f "${run_dir}/inference/checkpoints/model.pt" ]]; then
      continue
    fi
    case "${setting}" in
      profile) result_file="eval_profile.json" ;;
      fixed-2) result_file="eval_fixed_2f.json" ;;
      zero) result_file="eval_l0.json" ;;
    esac
    if [[ -f "${run_dir}/inference/${result_file}" ]]; then
      continue
    fi
    "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.evaluate_h8_checkpoint \
      --model "${model}" --run-dir "${run_dir}" \
      --vla-python "${vla_python}" --task-python "${MIKASA_PY_EXEC}" \
      --eval-cuda-visible-devices "${devices}" --setting "${setting}" \
      --inference-device "${inference_device}" --simulator-device "${simulator_device}" \
      --label "final-${setting}"
  done
}

evaluate_completed() {
  eval_model qwenoft "${QWENOFT_PY_EXEC}" 1 cuda:0 cuda:0
  eval_model qwenpi_v3 "${QWENPI_PY_EXEC}" 1 cuda:0 cuda:0
  eval_model qwengr00t "${QWENGROOT_PY_EXEC}" 1 cuda:0 cuda:0
}

eval_zero_fixed_model() {
  local model="$1"
  local vla_python="$2"
  local run_dir="${ZERO_BASELINE_ROOT}/${model}"
  if [[ -f "${run_dir}/inference/eval_fixed_2f.json" ]]; then
    return
  fi
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.evaluate_h8_checkpoint \
    --model "${model}" --run-dir "${run_dir}" \
    --vla-python "${vla_python}" --task-python "${MIKASA_PY_EXEC}" \
    --eval-cuda-visible-devices 1 --setting fixed-2 \
    --inference-device cuda:0 --simulator-device cuda:0 \
    --label final-fixed-2
}

evaluate_zero_fixed() {
  eval_zero_fixed_model qwenoft "${QWENOFT_PY_EXEC}"
  eval_zero_fixed_model qwenpi_v3 "${QWENPI_PY_EXEC}"
  eval_zero_fixed_model qwengr00t "${QWENGROOT_PY_EXEC}"
}

evaluate() {
  set -a
  . "${REPO_ROOT}/.env"
  set +a
  export WANDB_ENTITY="${MIKASA_WANDB_ENTITY:-placeholder}"
  while [[ -n "$(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    sleep 60
  done
  evaluate_completed
  evaluate_zero_fixed
  "${QWENOFT_PY_EXEC}" \
    -m scripts.tasks.mikasa_intercept_grab_fast.summarize_conditioned_h8 \
    --root "${OUTPUT_ROOT}" > "${OUTPUT_ROOT}/evaluation-report.log"
}

case "${PHASE}" in
  collect) collect ;;
  train) train ;;
  eval) evaluate ;;
  eval-completed) evaluate_completed ;;
  eval-zero-fixed) evaluate_zero_fixed ;;
  all) collect; train; evaluate ;;
  *) echo "Usage: $0 {collect|train|eval|eval-completed|eval-zero-fixed|all}" >&2; exit 2 ;;
esac
