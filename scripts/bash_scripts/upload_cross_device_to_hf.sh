#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-}
OUT=${OUT:-results/cross_device}
HF_REPO=${HF_REPO:-}
HF_REPO_TYPE=${HF_REPO_TYPE:-dataset}
HF_TOKEN=${HF_TOKEN:-}
HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-}
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-}
DEVICE=${DEVICE:-auto}
RUNTIME=${RUNTIME:-combined}
REMOTE_PREFIX=${REMOTE_PREFIX:-raw}
RUN_TAG=${RUN_TAG:-}
DRY_RUN=${DRY_RUN:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_root() {
    if [[ -n "${REPO_DIR}" ]]; then
        cd "${REPO_DIR}" && pwd
        return
    fi
    cd "${SCRIPT_DIR}/../.." && pwd
}

init_log_file() {
    local repo_root="$1"
    local log_dir="${repo_root}/logs/cross_device_protocol"
    mkdir -p "${log_dir}"
    LOG_FILE="${log_dir}/upload_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
}

detect_device_slug() {
    if [[ "${DEVICE}" != "auto" ]]; then
        echo "${DEVICE}"
        return
    fi

    local gpu_name=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_output=""
        if gpu_output="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; then
            gpu_name="$(printf '%s\n' "${gpu_output}" | head -n 1)"
        fi
    fi

    case "${gpu_name}" in
        *"RTX 3090"*) echo "rtx3090" ;;
        *"RTX 4090"*) echo "rtx4090" ;;
        *"RTX PRO 6000 Blackwell"*) echo "rtxpro6000blackwell" ;;
        *"L40S"*) echo "l40s" ;;
        *"A100"*) echo "a100" ;;
        *"H200"*) echo "h200" ;;
        *"B200"*) echo "b200" ;;
        "")
            echo "Error: DEVICE=auto but nvidia-smi did not report a GPU. Set DEVICE manually." >&2
            return 1
            ;;
        *)
            echo "Error: DEVICE=auto could not map GPU '${gpu_name}'. Set DEVICE manually, e.g. DEVICE=a100." >&2
            return 1
            ;;
    esac
}

find_hf_cli() {
    if command -v hf >/dev/null 2>&1; then
        echo "hf"
        return
    fi
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli"
        return
    fi
    echo "Error: neither 'hf' nor 'huggingface-cli' is available. Install huggingface_hub or activate an env that has it." >&2
    return 1
}

resolve_hf_token() {
    if [[ -n "${HF_TOKEN}" ]]; then
        printf '%s' "${HF_TOKEN}"
        return
    fi
    if [[ -n "${HUGGINGFACE_HUB_TOKEN}" ]]; then
        printf '%s' "${HUGGINGFACE_HUB_TOKEN}"
        return
    fi
    if [[ -n "${HUGGINGFACE_TOKEN}" ]]; then
        printf '%s' "${HUGGINGFACE_TOKEN}"
        return
    fi
    if [[ -n "${HF_TOKEN_FILE}" ]]; then
        if [[ ! -f "${HF_TOKEN_FILE}" ]]; then
            echo "Error: HF_TOKEN_FILE does not exist: ${HF_TOKEN_FILE}" >&2
            return 1
        fi
        head -n 1 "${HF_TOKEN_FILE}" | tr -d '\r\n'
        return
    fi
    return 0
}

REPO_ROOT="$(resolve_repo_root)"
cd "${REPO_ROOT}"
init_log_file "${REPO_ROOT}"

if [[ -z "${HF_REPO}" ]]; then
    echo "Error: set HF_REPO first, for example:" >&2
    echo "  HF_REPO=your-name/cross-device-latency-v1 bash scripts/bash_scripts/upload_cross_device_to_hf.sh" >&2
    exit 1
fi

if [[ ! -d "${OUT}" ]]; then
    echo "Error: OUT does not exist: ${OUT}" >&2
    exit 1
fi
if [[ ! -f "${OUT}/manifests/runs.jsonl" ]]; then
    echo "Error: missing manifest file: ${OUT}/manifests/runs.jsonl" >&2
    exit 1
fi

DEVICE_SLUG="$(detect_device_slug)"
if [[ -z "${RUN_TAG}" ]]; then
    RUN_TAG="$(hostname)_$(date +%Y%m%d_%H%M%S)"
fi
REMOTE_PATH="${REMOTE_PREFIX}/${DEVICE_SLUG}/${RUNTIME}/${RUN_TAG}/cross_device"
HF_CLI="$(find_hf_cli)"
RESOLVED_HF_TOKEN="$(resolve_hf_token)"
if [[ -n "${RESOLVED_HF_TOKEN}" ]]; then
    export HF_TOKEN="${RESOLVED_HF_TOKEN}"
    export HUGGINGFACE_HUB_TOKEN="${RESOLVED_HF_TOKEN}"
fi

echo "== Upload cross-device protocol outputs =="
echo "Repo root: ${REPO_ROOT}"
echo "Local OUT: ${OUT}"
echo "HF repo: ${HF_REPO}"
echo "HF repo type: ${HF_REPO_TYPE}"
if [[ -n "${RESOLVED_HF_TOKEN}" ]]; then
    echo "HF token: provided"
else
    echo "HF token: not provided; relying on existing hf login/cache"
fi
echo "Device: ${DEVICE_SLUG}"
echo "Runtime label: ${RUNTIME}"
echo "Remote path: ${REMOTE_PATH}"
echo "DRY_RUN: ${DRY_RUN}"
if command -v nvidia-smi >/dev/null 2>&1; then
    if GPU_OUTPUT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; then
        echo "Detected GPU(s):"
        printf '%s\n' "${GPU_OUTPUT}"
    else
        echo "Detected GPU(s): nvidia-smi is present but not usable"
    fi
fi
echo "Manifest lines: $(wc -l < "${OUT}/manifests/runs.jsonl")"
echo "Profile count: $(find "${OUT}/profiles" -name profile.json 2>/dev/null | wc -l || true)"
echo "Run manifest count: $(find "${OUT}/runs" -name manifest.json 2>/dev/null | wc -l || true)"

CMD=(
    "${HF_CLI}" upload
    --repo-type "${HF_REPO_TYPE}"
    "${HF_REPO}"
    "${OUT}"
    "${REMOTE_PATH}"
)

echo "Command:"
printf '  %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run only; not uploading."
    echo "Log file: ${LOG_FILE}"
    exit 0
fi

"${CMD[@]}"

echo "== upload complete =="
echo "Uploaded to: ${HF_REPO}/${REMOTE_PATH}"
echo "Log file: ${LOG_FILE}"
