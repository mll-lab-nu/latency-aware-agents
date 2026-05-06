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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/bash_scripts/cross_device_launcher_common.sh
source "${SCRIPT_DIR}/cross_device_launcher_common.sh"

REPO_ROOT="$(cross_device_resolve_repo_root "${BASH_SOURCE[0]}")"
cd "${REPO_ROOT}"
cross_device_init_logging "${REPO_ROOT}" "aggregate"
cross_device_activate_env "latency"
cross_device_verify_protocol_script
cross_device_optional_args

echo "== Cross-device aggregate =="
cross_device_print_environment
echo "Spec: ${SPEC}"
echo "Output root: ${OUT}"
echo "DRY_RUN: ${DRY_RUN}"

python scripts/experiments/run_cross_device_protocol.py aggregate \
    --spec "${SPEC}" \
    --root "${OUT}" \
    --out "${OUT}/tables" \
    --overwrite \
    "${OPTIONAL_ARGS[@]}"

echo "CSV tables:"
echo "${OUT}/tables/experiment1_latency_profile_summary.csv"
echo "${OUT}/tables/experiment2_same_device_calibration.csv"
echo "${OUT}/tables/experiment3_cross_device_prediction.csv"
echo "Log file: ${LOG_FILE}"
