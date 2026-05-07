#!/usr/bin/env bash

cross_device_resolve_repo_root() {
    local script_path="$1"
    if [[ -n "${REPO_DIR:-}" ]]; then
        if [[ ! -d "${REPO_DIR}" ]]; then
            echo "Error: REPO_DIR does not exist: ${REPO_DIR}" >&2
            return 1
        fi
        cd "${REPO_DIR}" && pwd
        return
    fi

    local script_dir
    script_dir="$(cd "$(dirname "${script_path}")" && pwd)"
    cd "${script_dir}/../.." && pwd
}

cross_device_source_conda() {
    local candidates=()
    if [[ -n "${CONDA_EXE:-}" ]]; then
        candidates+=("$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh")
    fi
    candidates+=(
        "${HOME}/miniconda3/etc/profile.d/conda.sh"
        "${HOME}/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            # shellcheck source=/dev/null
            source "${candidate}"
            break
        fi
    done

    if ! type conda >/dev/null 2>&1; then
        echo "Error: conda is not available. Set CONDA_EXE or install miniconda/anaconda." >&2
        return 1
    fi
}

cross_device_ensure_conda_env() {
    local env_name="$1"
    if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${env_name}"; then
        echo "Error: conda env '${env_name}' was not found." >&2
        echo "Available conda envs:" >&2
        conda env list >&2
        return 1
    fi
}

cross_device_activate_env() {
    local env_name="$1"
    cross_device_source_conda
    cross_device_ensure_conda_env "${env_name}"
    conda activate "${env_name}"
}

cross_device_init_logging() {
    local repo_root="$1"
    local label="$2"
    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local log_dir="${repo_root}/logs/cross_device_protocol"
    mkdir -p "${log_dir}"
    LOG_FILE="${log_dir}/${label}_${timestamp}.log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "Log file: ${LOG_FILE}"
}

cross_device_optional_args() {
    OPTIONAL_ARGS=()
    if [[ -n "${N_EPISODES:-}" ]]; then
        OPTIONAL_ARGS+=(--n-episodes "${N_EPISODES}")
    fi
    if [[ -n "${SEED_SET:-}" ]]; then
        OPTIONAL_ARGS+=(--seed-set "${SEED_SET}")
    fi
    if [[ -n "${CHECKPOINT_KIND:-}" ]]; then
        OPTIONAL_ARGS+=(--checkpoint-kind "${CHECKPOINT_KIND}")
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        OPTIONAL_ARGS+=(--dry-run)
    fi
    if [[ -n "${ONLY_ENV:-}" ]]; then
        OPTIONAL_ARGS+=(--only-env "${ONLY_ENV}")
    fi
    if [[ -n "${ONLY_POLICY:-}" ]]; then
        OPTIONAL_ARGS+=(--only-policy "${ONLY_POLICY}")
    fi
    if [[ -n "${ONLY_DEVICE:-}" ]]; then
        OPTIONAL_ARGS+=(--only-device "${ONLY_DEVICE}")
    fi
    if [[ -n "${ONLY_EXPERIMENT:-}" ]]; then
        OPTIONAL_ARGS+=(--only-experiment "${ONLY_EXPERIMENT}")
    fi
    if [[ -n "${MAX_RUNS:-}" ]]; then
        OPTIONAL_ARGS+=(--max-runs "${MAX_RUNS}")
    fi
    EXECUTION_OPTIONAL_ARGS=("${OPTIONAL_ARGS[@]}")
    if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
        EXECUTION_OPTIONAL_ARGS+=(--continue-on-error)
    fi
    if [[ "${OPENVLA_SMOKE:-0}" == "1" ]]; then
        EXECUTION_OPTIONAL_ARGS+=(--smoke --smoke-episodes "${OPENVLA_SMOKE_EPISODES:-1}")
        if [[ -n "${OPENVLA_SMOKE_MAX_STEPS:-}" ]]; then
            EXECUTION_OPTIONAL_ARGS+=(--smoke-max-steps "${OPENVLA_SMOKE_MAX_STEPS}")
        fi
    fi
}

cross_device_openvla_resolve_stages() {
    OPENVLA_SELECTED_STAGES=()
    local raw_stages="${OPENVLA_STAGES:-}"
    if [[ -z "${raw_stages}" ]]; then
        if [[ "${OPENVLA_SMOKE:-0}" == "1" ]]; then
            OPENVLA_SELECTED_STAGES=(doctor measure realtime)
        else
            OPENVLA_SELECTED_STAGES=(doctor measure realtime sim-same sim-cross)
        fi
        return 0
    fi

    local IFS=,
    local raw stage
    read -ra OPENVLA_SELECTED_STAGES <<<"${raw_stages}"
    for i in "${!OPENVLA_SELECTED_STAGES[@]}"; do
        raw="${OPENVLA_SELECTED_STAGES[$i]}"
        stage="$(echo "${raw}" | xargs)"
        stage="${stage/_/-}"
        case "${stage}" in
            doctor|measure|realtime|sim-same|sim-cross)
                OPENVLA_SELECTED_STAGES[$i]="${stage}"
                ;;
            *)
                echo "Error: unsupported OPENVLA_STAGES entry '${raw}'. Use doctor,measure,realtime,sim-same,sim-cross." >&2
                return 2
                ;;
        esac
    done
}

cross_device_openvla_stage_selected() {
    local wanted="$1"
    local stage
    for stage in "${OPENVLA_SELECTED_STAGES[@]}"; do
        if [[ "${stage}" == "${wanted}" ]]; then
            return 0
        fi
    done
    return 1
}

cross_device_reset_failures() {
    CROSS_DEVICE_FAILED_STAGES=()
}

cross_device_run_protocol_stage() {
    local stage_name="$1"
    shift
    echo "== ${stage_name} =="
    local errexit_was_set=0
    if [[ "$-" == *e* ]]; then
        errexit_was_set=1
        set +e
    fi
    "$@"
    local status=$?
    if [[ "${errexit_was_set}" -eq 1 ]]; then
        set -e
    fi
    if [[ "${status}" -eq 0 ]]; then
        return 0
    fi
    if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
        echo "Stage failed but CONTINUE_ON_ERROR=1, continuing: ${stage_name} (exit ${status})" >&2
        CROSS_DEVICE_FAILED_STAGES+=("${stage_name}|${status}|$*")
        return 0
    fi
    return "${status}"
}

cross_device_finish_protocol_runs() {
    if [[ ${#CROSS_DEVICE_FAILED_STAGES[@]} -eq 0 ]]; then
        return 0
    fi
    echo "FAILED STAGES:" >&2
    local item stage status command
    for item in "${CROSS_DEVICE_FAILED_STAGES[@]}"; do
        IFS='|' read -r stage status command <<<"${item}"
        echo "  - stage=${stage} exit_code=${status} command=${command}" >&2
    done
    return 1
}

cross_device_print_environment() {
    echo "Repo root: $(pwd)"
    echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-}"
    echo "Python path: $(which python)"
    echo "Python version: $(python --version 2>&1)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "Detected GPU(s):"
        nvidia-smi --query-gpu=name --format=csv,noheader || true
    else
        echo "nvidia-smi: not found"
    fi
}

cross_device_verify_protocol_script() {
    if [[ ! -f "scripts/experiments/run_cross_device_protocol.py" ]]; then
        echo "Error: missing scripts/experiments/run_cross_device_protocol.py in $(pwd)" >&2
        return 1
    fi
}

cross_device_print_result_locations() {
    local out="$1"
    echo "Results root: ${out}"
    echo "Profiles: ${out}/profiles"
    echo "Runs: ${out}/runs"
    echo "Manifests: ${out}/manifests/runs.jsonl"
    echo "Tables: ${out}/tables"
}
