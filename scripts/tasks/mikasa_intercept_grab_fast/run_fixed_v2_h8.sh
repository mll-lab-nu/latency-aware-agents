#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:?Usage: run_fixed_v2_h8.sh collect|train-eval|verify|all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-v2"
OUTPUT_ROOT="${REPO_ROOT}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned"
TEACHER="${REPO_ROOT}/outputs/starvla/artifacts/models/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt"
PACKAGER_PROJECT="${REPO_ROOT}/third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot"
RUN_ID="h8-conditioned-seed0-fixed-2"
QWENOFT_PY_EXEC="${QWENOFT_PY:?Set QWENOFT_PY}"
QWENPI_PY_EXEC="${QWENPI_PY:?Set QWENPI_PY}"
QWENGROOT_PY_EXEC="${QWENGROOT_PY:?Set QWENGROOT_PY}"
MIKASA_PY_EXEC="${MIKASA_PY:?Set MIKASA_PY}"
UV_EXEC="${UV:?Set UV}"
FIXED_COLLECT_GPU="${FIXED_COLLECT_GPU:-0}"

cd "${REPO_ROOT}"

collect() {
  mkdir -p "${DATA_ROOT}/configs"
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.write_conditioned_latency_config \
    --condition fixed-2 --output "${DATA_ROOT}/configs/fixed-2.yaml"
  CUDA_VISIBLE_DEVICES="${FIXED_COLLECT_GPU}" PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/MIKASA-Robo" \
    "${MIKASA_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 \
    --condition fixed-2 --model shared \
    --latency-config "${DATA_ROOT}/configs/fixed-2.yaml" \
    --checkpoint "${TEACHER}" --output "${DATA_ROOT}/fixed-2/raw" \
    --train-targets 64288 --validation-targets 6884 --num-envs 128 \
    > "${DATA_ROOT}/fixed-2-collect.log" 2>&1
  PYTHONPATH="${REPO_ROOT}" "${UV_EXEC}" run --frozen --project "${PACKAGER_PROJECT}" python \
    -m scripts.tasks.mikasa_intercept_grab_fast.prepare_conditioned_h8_dataset \
    --raw-root "${DATA_ROOT}/fixed-2/raw" --output-train "${DATA_ROOT}/fixed-2/lerobot"
  PYTHONPATH="${REPO_ROOT}" "${UV_EXEC}" run --frozen --project "${PACKAGER_PROJECT}" python \
    -m scripts.tasks.mikasa_intercept_grab_fast.audit_conditioned_h8_dataset \
    --condition fixed-2 --raw-root "${DATA_ROOT}/fixed-2/raw" \
    --dataset "${DATA_ROOT}/fixed-2/lerobot"
}

train_eval_model() {
  local model="$1"
  local vla_python="$2"
  local devices="$3"
  local port="$4"
  local output="${OUTPUT_ROOT}/fixed-2/${model}"
  CUDA_VISIBLE_DEVICES="${devices}" VLA_PY="${vla_python}" \
    bash scripts/tasks/mikasa_intercept_grab_fast/train_conditioned_h8.sh "${model}" \
    --condition fixed-2 --train-only --no-oom-retry --gradient-checkpointing \
    --run-id "${RUN_ID}" --output-dir "${output}" \
    --dataset-dir "${DATA_ROOT}/fixed-2" \
    --num-gpus 2 --micro-batch 64 --port "${port}" --task-python "${MIKASA_PY_EXEC}" \
    > "${OUTPUT_ROOT}/fixed-${model}-v2.log" 2>&1
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.evaluate_h8_checkpoint \
    --model "${model}" --run-dir "${output}" --vla-python "${vla_python}" \
    --task-python "${MIKASA_PY_EXEC}" --eval-cuda-visible-devices "${devices}" \
    --setting fixed-2 --inference-device cuda:1 --simulator-device cuda:0 \
    --seed 4242424242 --episodes 50 --label final-fixed-2
}

train_eval() {
  train_eval_model qwenoft "${QWENOFT_PY_EXEC}" 0,1 29530 & P1=$!
  train_eval_model qwenpi_v3 "${QWENPI_PY_EXEC}" 2,3 29532 & P2=$!
  train_eval_model qwengr00t "${QWENGROOT_PY_EXEC}" 4,5 29534 & P3=$!
  failed=0
  for pid in "${P1}" "${P2}" "${P3}"; do wait "${pid}" || failed=1; done
  return "${failed}"
}

verify() {
  "${QWENOFT_PY_EXEC}" -m scripts.tasks.mikasa_intercept_grab_fast.verify_fixed_v2_h8 \
    --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}"
}

case "${PHASE}" in
  collect)
    rm -rf "${DATA_ROOT}/fixed-2"
    collect
    ;;
  train-eval)
    rm -rf \
      "${OUTPUT_ROOT}/fixed-2/qwenoft" \
      "${OUTPUT_ROOT}/fixed-2/qwenpi_v3" \
      "${OUTPUT_ROOT}/fixed-2/qwengr00t"
    train_eval
    ;;
  verify) verify ;;
  all)
    rm -rf "${DATA_ROOT}/fixed-2"
    collect
    rm -rf \
      "${OUTPUT_ROOT}/fixed-2/qwenoft" \
      "${OUTPUT_ROOT}/fixed-2/qwenpi_v3" \
      "${OUTPUT_ROOT}/fixed-2/qwengr00t"
    train_eval
    verify
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    exit 2
    ;;
esac
