#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-}
SPEC=${SPEC:-configs/experiments/cross_device_protocol.yaml}
OUT=${OUT:-results/cross_device}
RAW_ROOT=${RAW_ROOT:-results/raw}
USE_RAW=${USE_RAW:-auto}
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
echo "Raw root: ${RAW_ROOT}"
echo "USE_RAW: ${USE_RAW}"
echo "DRY_RUN: ${DRY_RUN}"

count_raw_manifests() {
    local root="$1"
    if [[ ! -d "${root}" ]]; then
        echo 0
        return
    fi
    find "${root}" -path '*/cross_device/manifests/runs.jsonl' -type f | wc -l
}

RAW_MANIFEST_COUNT=0
RAW_ROOT_EFFECTIVE="${RAW_ROOT}"
RAW_MANIFEST_COUNT="$(count_raw_manifests "${RAW_ROOT_EFFECTIVE}")"
if [[ "${RAW_MANIFEST_COUNT}" -eq 0 && -d "${RAW_ROOT}/raw" ]]; then
    RAW_NESTED_COUNT="$(count_raw_manifests "${RAW_ROOT}/raw")"
    if [[ "${RAW_NESTED_COUNT}" -gt 0 ]]; then
        RAW_ROOT_EFFECTIVE="${RAW_ROOT}/raw"
        RAW_MANIFEST_COUNT="${RAW_NESTED_COUNT}"
    fi
fi

USE_RAW_EFFECTIVE=0
case "${USE_RAW}" in
    auto)
        if [[ "${RAW_MANIFEST_COUNT}" -gt 0 ]]; then
            USE_RAW_EFFECTIVE=1
        fi
        ;;
    1|true|TRUE|yes|YES)
        if [[ "${RAW_MANIFEST_COUNT}" -eq 0 ]]; then
            echo "Error: USE_RAW=${USE_RAW} but no raw manifests found under ${RAW_ROOT}" >&2
            exit 1
        fi
        USE_RAW_EFFECTIVE=1
        ;;
    0|false|FALSE|no|NO)
        USE_RAW_EFFECTIVE=0
        ;;
    *)
        echo "Error: USE_RAW must be auto, 1, or 0; got '${USE_RAW}'" >&2
        exit 1
        ;;
esac

AGGREGATE_ARGS=(
    scripts/experiments/run_cross_device_protocol.py
    aggregate
    --spec "${SPEC}"
    --root "${OUT}"
    --out "${OUT}/tables"
    --overwrite
)
if [[ "${USE_RAW_EFFECTIVE}" -eq 1 ]]; then
    AGGREGATE_ARGS+=(--raw-root "${RAW_ROOT_EFFECTIVE}")
    echo "Raw aggregate mode: enabled (${RAW_MANIFEST_COUNT} manifest file(s), root=${RAW_ROOT_EFFECTIVE})"
else
    echo "Raw aggregate mode: disabled"
fi

python "${AGGREGATE_ARGS[@]}" "${OPTIONAL_ARGS[@]}"

echo "CSV tables:"
echo "${OUT}/tables/experiment1_latency_profile_summary.csv"
echo "${OUT}/tables/experiment2_same_device_calibration.csv"
echo "${OUT}/tables/experiment3_cross_device_prediction.csv"
echo "Log file: ${LOG_FILE}"
