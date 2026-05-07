#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-}
SPEC=${SPEC:-configs/experiments/cross_device_protocol.yaml}
OUT=${OUT:-results/cross_device}
PROFILES_DIR=${PROFILES_DIR:-results/cross_device/profiles}
N_EPISODES=${N_EPISODES:-}
SEED_SET=${SEED_SET:-}
CHECKPOINT_KIND=${CHECKPOINT_KIND:-}
DRY_RUN=${DRY_RUN:-0}
RUNTIME_ENV=${RUNTIME_ENV:-openvla}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/bash_scripts/cross_device_launcher_common.sh
source "${SCRIPT_DIR}/cross_device_launcher_common.sh"

REPO_ROOT="$(cross_device_resolve_repo_root "${BASH_SOURCE[0]}")"
cd "${REPO_ROOT}"
cross_device_init_logging "${REPO_ROOT}" "openvla"
cross_device_activate_env "latency-openvla"
cross_device_verify_protocol_script
cross_device_optional_args
cross_device_reset_failures

export CROSS_DEVICE_RUNTIME_ENV="latency-openvla"

echo "== Cross-device OpenVLA protocol =="
cross_device_print_environment
echo "Spec: ${SPEC}"
echo "Output root: ${OUT}"
echo "Profiles dir: ${PROFILES_DIR}"
echo "DRY_RUN: ${DRY_RUN}"
echo "RUNTIME_ENV: ${RUNTIME_ENV}"
echo "ONLY_POLICY: ${ONLY_POLICY:-}"
echo "ONLY_ENV: ${ONLY_ENV:-}"
echo "ONLY_DEVICE: ${ONLY_DEVICE:-}"
echo "ONLY_EXPERIMENT: ${ONLY_EXPERIMENT:-}"
echo "MAX_RUNS: ${MAX_RUNS:-}"
echo "CONTINUE_ON_ERROR: ${CONTINUE_ON_ERROR:-0}"
echo "OPENVLA_BASE_MODEL_PATH: ${OPENVLA_BASE_MODEL_PATH:-}"
echo "DEMON_ATTACK_OPENVLA_CHECKPOINT_PATH: ${DEMON_ATTACK_OPENVLA_CHECKPOINT_PATH:-}"
echo "DEADLY_CORRIDOR_OPENVLA_CHECKPOINT_PATH: ${DEADLY_CORRIDOR_OPENVLA_CHECKPOINT_PATH:-}"

cross_device_run_protocol_stage doctor \
python scripts/experiments/run_cross_device_protocol.py doctor \
    --spec "${SPEC}" \
    --hardware auto \
    --runtime-env "${RUNTIME_ENV}" \
    --out "${OUT}" \
    --profiles-dir "${PROFILES_DIR}" \
    "${OPTIONAL_ARGS[@]}"

cross_device_run_protocol_stage measure \
python scripts/experiments/run_cross_device_protocol.py measure \
    --spec "${SPEC}" \
    --hardware auto \
    --runtime-env "${RUNTIME_ENV}" \
    --out "${OUT}" \
    --resume \
    "${OPTIONAL_ARGS[@]}"

cross_device_run_protocol_stage realtime \
python scripts/experiments/run_cross_device_protocol.py realtime \
    --spec "${SPEC}" \
    --hardware auto \
    --runtime-env "${RUNTIME_ENV}" \
    --out "${OUT}" \
    --resume \
    "${OPTIONAL_ARGS[@]}"

cross_device_run_protocol_stage sim-cross \
python scripts/experiments/run_cross_device_protocol.py sim-cross \
    --spec "${SPEC}" \
    --replay-hardware auto \
    --runtime-env "${RUNTIME_ENV}" \
    --profiles-dir "${PROFILES_DIR}" \
    --out "${OUT}" \
    --resume \
    "${OPTIONAL_ARGS[@]}"

echo "If sim-cross selected no runnable rows, check the doctor output above for skip reasons."
echo "== complete =="
cross_device_print_result_locations "${OUT}"
echo "Log file: ${LOG_FILE}"
cross_device_finish_protocol_runs
