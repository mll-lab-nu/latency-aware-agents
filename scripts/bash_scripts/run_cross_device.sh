#!/usr/bin/env bash
set -euo pipefail

# Minimal cross-device protocol entrypoint.
#
# Common usage:
#   STAGES="doctor measure realtime sim-same sim-cross aggregate" scripts/bash_scripts/run_cross_device.sh
#   DRY_RUN=1 ONLY_POLICY=OpenVLA-Flappy STAGES="doctor measure" scripts/bash_scripts/run_cross_device.sh
#   HF_PROFILE_REPO=latency-sensitive-bench/profiles STAGES="sim-same sim-cross" scripts/bash_scripts/run_cross_device.sh
#   UPLOAD_PROFILES=1 HF_PROFILE_REPO=latency-sensitive-bench/profiles STAGES="measure" scripts/bash_scripts/run_cross_device.sh
#
# Key knobs:
#   SPEC, OUT, PROFILES_DIR, TABLES_DIR
#   HARDWARE, REPLAY_HARDWARE, RUNTIME_ENV
#   STAGES: space-separated list from doctor, measure, realtime, sim-same, sim-cross, aggregate
#   ONLY_POLICY, ONLY_ENV, ONLY_DEVICE, ONLY_EXPERIMENT, MAX_RUNS
#   N_EPISODES, SEED_SET, CHECKPOINT_KIND
#   DRY_RUN=1, RESUME=1, OVERWRITE=1, CONTINUE_ON_ERROR=1, SMOKE=1
#   HF_PROFILE_REPO: dataset repo used by sim stages to download missing profiles
#   UPLOAD_PROFILES=1: upload local profile folders after selected stages finish

SPEC=${SPEC:-configs/experiments/cross_device_protocol.yaml}
OUT=${OUT:-results/cross_device}
PROFILES_DIR=${PROFILES_DIR:-${OUT}/profiles}
TABLES_DIR=${TABLES_DIR:-${OUT}/tables}
HARDWARE=${HARDWARE:-auto}
REPLAY_HARDWARE=${REPLAY_HARDWARE:-auto}
RUNTIME_ENV=${RUNTIME_ENV:-auto}
STAGES=${STAGES:-doctor measure realtime sim-same sim-cross aggregate}
DRY_RUN=${DRY_RUN:-0}
RESUME=${RESUME:-1}
OVERWRITE=${OVERWRITE:-0}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}
SMOKE=${SMOKE:-0}
UPLOAD_PROFILES=${UPLOAD_PROFILES:-0}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

COMMON_ARGS=(--spec "${SPEC}" --runtime-env "${RUNTIME_ENV}")
CHECK_ARGS=()
EXEC_ARGS=()
AGG_ARGS=(--spec "${SPEC}" --root "${OUT}" --out "${TABLES_DIR}")

if [[ -n "${ONLY_POLICY:-}" ]]; then COMMON_ARGS+=(--only-policy "${ONLY_POLICY}"); AGG_ARGS+=(--only-policy "${ONLY_POLICY}"); fi
if [[ -n "${ONLY_ENV:-}" ]]; then COMMON_ARGS+=(--only-env "${ONLY_ENV}"); AGG_ARGS+=(--only-env "${ONLY_ENV}"); fi
if [[ -n "${ONLY_DEVICE:-}" ]]; then COMMON_ARGS+=(--only-device "${ONLY_DEVICE}"); AGG_ARGS+=(--only-device "${ONLY_DEVICE}"); fi
if [[ -n "${ONLY_EXPERIMENT:-}" ]]; then COMMON_ARGS+=(--only-experiment "${ONLY_EXPERIMENT}"); AGG_ARGS+=(--only-experiment "${ONLY_EXPERIMENT}"); fi
if [[ -n "${MAX_RUNS:-}" ]]; then COMMON_ARGS+=(--max-runs "${MAX_RUNS}"); fi
if [[ -n "${N_EPISODES:-}" ]]; then COMMON_ARGS+=(--n-episodes "${N_EPISODES}"); fi
if [[ -n "${SEED_SET:-}" ]]; then COMMON_ARGS+=(--seed-set "${SEED_SET}"); fi
if [[ -n "${CHECKPOINT_KIND:-}" ]]; then COMMON_ARGS+=(--checkpoint-kind "${CHECKPOINT_KIND}"); fi
if [[ "${DRY_RUN}" == "1" ]]; then CHECK_ARGS+=(--dry-run); EXEC_ARGS+=(--dry-run); AGG_ARGS+=(--dry-run); fi
if [[ "${RESUME}" == "1" ]]; then EXEC_ARGS+=(--resume); fi
if [[ "${OVERWRITE}" == "1" ]]; then EXEC_ARGS+=(--overwrite); AGG_ARGS+=(--overwrite); fi
if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then EXEC_ARGS+=(--continue-on-error); fi
if [[ "${SMOKE}" == "1" ]]; then EXEC_ARGS+=(--smoke); fi
if [[ -n "${SMOKE_EPISODES:-}" ]]; then EXEC_ARGS+=(--smoke-episodes "${SMOKE_EPISODES}"); fi
if [[ -n "${SMOKE_MAX_STEPS:-}" ]]; then EXEC_ARGS+=(--smoke-max-steps "${SMOKE_MAX_STEPS}"); fi

echo "Cross-device protocol"
echo "  spec: ${SPEC}"
echo "  out: ${OUT}"
echo "  profiles: ${PROFILES_DIR}"
echo "  stages: ${STAGES}"
echo "  runtime_env: ${RUNTIME_ENV}"
echo "  hf_profile_repo: ${HF_PROFILE_REPO:-}"

for stage in ${STAGES}; do
    case "${stage}" in
        doctor)
            python -m scripts.cross_device.protocol doctor \
                "${COMMON_ARGS[@]}" --hardware "${HARDWARE}" --out "${OUT}" --profiles-dir "${PROFILES_DIR}" "${CHECK_ARGS[@]}"
            ;;
        check-deps)
            python -m scripts.cross_device.protocol check-deps \
                "${COMMON_ARGS[@]}" --hardware "${HARDWARE}" --out "${OUT}" --profiles-dir "${PROFILES_DIR}" "${CHECK_ARGS[@]}"
            ;;
        measure)
            python -m scripts.cross_device.protocol measure \
                "${COMMON_ARGS[@]}" --hardware "${HARDWARE}" --out "${OUT}" "${EXEC_ARGS[@]}"
            ;;
        realtime)
            python -m scripts.cross_device.protocol realtime \
                "${COMMON_ARGS[@]}" --hardware "${HARDWARE}" --out "${OUT}" "${EXEC_ARGS[@]}"
            ;;
        sim-same)
            python -m scripts.cross_device.protocol sim-same \
                "${COMMON_ARGS[@]}" --hardware "${HARDWARE}" --profiles-dir "${PROFILES_DIR}" --out "${OUT}" "${EXEC_ARGS[@]}"
            ;;
        sim-cross)
            python -m scripts.cross_device.protocol sim-cross \
                "${COMMON_ARGS[@]}" --replay-hardware "${REPLAY_HARDWARE}" --profiles-dir "${PROFILES_DIR}" --out "${OUT}" "${EXEC_ARGS[@]}"
            ;;
        aggregate)
            python -m scripts.cross_device.protocol aggregate "${AGG_ARGS[@]}"
            ;;
        *)
            echo "Unknown stage: ${stage}" >&2
            exit 2
            ;;
    esac
done

if [[ "${UPLOAD_PROFILES}" == "1" ]]; then
    : "${HF_PROFILE_REPO:?Set HF_PROFILE_REPO before uploading profiles}"
    while IFS= read -r profile_json; do
        python -m scripts.cross_device.profile_cache upload \
            --profile-path "${profile_json}" \
            --profiles-dir "${PROFILES_DIR}" \
            --hf-repo "${HF_PROFILE_REPO}"
    done < <(find "${PROFILES_DIR}" -name profile.json)
fi
