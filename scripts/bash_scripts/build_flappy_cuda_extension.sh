#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CUDA_TOOLKIT_SPEC="${LATENCY_BENCH_CUDA_TOOLKIT_SPEC:-cuda-toolkit[nvcc,cccl]==13.0.*}"

detect_gpu_compute_cap() {
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '[:space:]'
}

target_sm_code() {
    local compute_cap="$1"
    printf 'sm_%s\n' "${compute_cap/./}"
}

collect_system_nvccs() {
    printf '%s\n' /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc 2>/dev/null | awk '!seen[$0]++'
}

collect_env_nvccs() {
    local env_prefix="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"
    if [[ -z "${env_prefix}" || ! -d "${env_prefix}" ]]; then
        return 0
    fi
    find "${env_prefix}" -type f -path '*/bin/nvcc' 2>/dev/null | awk '!seen[$0]++'
}

nvcc_supports_sm() {
    local nvcc_path="$1"
    local target_sm="$2"
    "${nvcc_path}" --list-gpu-code 2>/dev/null | tr ',[:space:]' '\n' | grep -qx "${target_sm}"
}

select_compatible_nvcc() {
    local target_sm="$1"
    shift
    local nvcc_path
    for nvcc_path in "$@"; do
        [[ -x "${nvcc_path}" ]] || continue
        if nvcc_supports_sm "${nvcc_path}" "${target_sm}"; then
            printf '%s\n' "${nvcc_path}"
            return 0
        fi
    done
    return 1
}

activate_cuda_toolkit() {
    local nvcc_path="$1"
    export CUDA_HOME
    CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    local cuda_lib_dirs=()
    if [[ -d "${CUDA_HOME}/lib64" ]]; then
        cuda_lib_dirs+=("${CUDA_HOME}/lib64")
    fi
    if [[ -d "${CUDA_HOME}/lib" ]]; then
        cuda_lib_dirs+=("${CUDA_HOME}/lib")
    fi
    if ((${#cuda_lib_dirs[@]} > 0)); then
        local joined_cuda_lib_dirs
        joined_cuda_lib_dirs="$(IFS=:; echo "${cuda_lib_dirs[*]}")"
        export LD_LIBRARY_PATH="${joined_cuda_lib_dirs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export LIBRARY_PATH="${joined_cuda_lib_dirs}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    fi
    hash -r
    echo "cuda_home=${CUDA_HOME}"
    echo "nvcc=${nvcc_path}"
    "${nvcc_path}" --version
}

ensure_cudart_link_compat() {
    local cuda_lib_dir
    for cuda_lib_dir in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib"; do
        [[ -d "${cuda_lib_dir}" ]] || continue
        if [[ -f "${cuda_lib_dir}/libcudart.so" ]]; then
            return 0
        fi
        local versioned_cudart
        versioned_cudart="$(find "${cuda_lib_dir}" -maxdepth 1 -type f -name 'libcudart.so.*' | sort -V | tail -n1)"
        if [[ -n "${versioned_cudart}" ]]; then
            local compat_dir="${REPO_ROOT}/.cache/flappy_cuda_link_compat"
            mkdir -p "${compat_dir}"
            ln -sfn "${versioned_cudart}" "${compat_dir}/libcudart.so"
            export LD_LIBRARY_PATH="${compat_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            export LIBRARY_PATH="${compat_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
            echo "Created temporary libcudart.so link in ${compat_dir}"
            return 0
        fi
    done
    return 0
}

ensure_compatible_nvcc() {
    local compute_cap
    compute_cap="$(detect_gpu_compute_cap)"
    local target_sm
    target_sm="$(target_sm_code "${compute_cap}")"
    echo "gpu_compute_cap=${compute_cap}"
    echo "target_sm=${target_sm}"

    mapfile -t system_nvccs < <(collect_system_nvccs)
    local selected_nvcc=""
    if ((${#system_nvccs[@]} > 0)); then
        selected_nvcc="$(select_compatible_nvcc "${target_sm}" "${system_nvccs[@]}" || true)"
    fi
    if [[ -n "${selected_nvcc}" ]]; then
        echo "Using system nvcc that supports ${target_sm}"
        activate_cuda_toolkit "${selected_nvcc}"
        ensure_cudart_link_compat
        return 0
    fi

    local env_prefix="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"
    if [[ -z "${env_prefix}" ]]; then
        echo "No compatible system nvcc found for ${target_sm} and no active environment prefix is available for toolkit install." >&2
        printf 'Scanned system nvcc paths:\n%s\n' "${system_nvccs[*]}" >&2
        return 1
    fi

    echo "No system nvcc supports ${target_sm}; installing ${CUDA_TOOLKIT_SPEC} into ${env_prefix}"
    python -m pip install "${CUDA_TOOLKIT_SPEC}"

    mapfile -t env_nvccs < <(collect_env_nvccs)
    if ((${#env_nvccs[@]} > 0)); then
        selected_nvcc="$(select_compatible_nvcc "${target_sm}" "${env_nvccs[@]}" || true)"
    else
        selected_nvcc=""
    fi
    if [[ -n "${selected_nvcc}" ]]; then
        echo "Using environment nvcc that supports ${target_sm}"
        activate_cuda_toolkit "${selected_nvcc}"
        ensure_cudart_link_compat
        return 0
    fi

    printf 'System nvcc paths scanned:\n' >&2
    printf '%s\n' "${system_nvccs[@]}" >&2
    printf 'Environment nvcc paths scanned:\n' >&2
    printf '%s\n' "${env_nvccs[@]}" >&2
    echo "Unable to find an nvcc that supports ${target_sm} after installing ${CUDA_TOOLKIT_SPEC}." >&2
    return 1
}

ensure_compatible_nvcc

python - <<'PY'
import sys

import torch

from training.flappy_sf.flappy_cuda_extension import load_flappy_cuda_extension

print(f"python={sys.executable}", flush=True)
print(f"torch={torch.__version__}", flush=True)
extension = load_flappy_cuda_extension()
torch.cuda.synchronize()
print(f"flappy_cuda_extension={extension.__name__}", flush=True)
PY
