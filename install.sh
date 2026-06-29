#!/usr/bin/env bash
set -euo pipefail

LATENCY_BENCH_TORCH_PROFILE="${LATENCY_BENCH_TORCH_PROFILE:-auto}"
MINICONDA_HOME="${HOME}/miniconda3"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
CONDA_SH_PATH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/bash_scripts/_pip.sh
source "${SCRIPT_DIR}/scripts/bash_scripts/_pip.sh"
# shellcheck source=scripts/bash_scripts/_mirrors.sh
source "${SCRIPT_DIR}/scripts/bash_scripts/_mirrors.sh"

detect_torch_profile() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[install] nvidia-smi not found; defaulting auto torch profile to cu126" >&2
        echo "cu126"
        return
    fi

    local compute_caps
    compute_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
    if [[ -z "${compute_caps//[[:space:]]/}" ]]; then
        echo "[install] unable to query GPU compute capability; defaulting auto torch profile to cu126" >&2
        echo "cu126"
        return
    fi

    if echo "${compute_caps}" | awk -F. '$1 >= 10 {found=1} END {exit found ? 0 : 1}'; then
        echo "[install] detected GPU compute capability >= 10.0; selecting cu130" >&2
        echo "cu130"
        return
    fi

    echo "[install] detected GPU compute capability < 10.0; selecting cu126" >&2
    echo "cu126"
}

resolve_torch_profile() {
    case "$1" in
        auto)
            detect_torch_profile
            ;;
        cu126|cu128|cu130)
            echo "$1"
            ;;
        *)
            echo "[install] Unknown LATENCY_BENCH_TORCH_PROFILE='$1'. Expected auto|cu126|cu128|cu130." >&2
            return 1
            ;;
    esac
}

resolve_torch_specs() {
    case "$1" in
        cu130)
            echo "torch==2.12.1+cu130 torchvision==0.27.1+cu130"
            ;;
        cu128)
            echo "torch==2.10.0+cu128 torchvision==0.25.0+cu128"
            ;;
        cu126)
            echo "torch==2.10.0+cu126 torchvision==0.25.0+cu126"
            ;;
        *)
            echo "[install] Unknown resolved torch profile '$1'. Expected cu126|cu128|cu130." >&2
            return 1
            ;;
    esac
}

find_conda_sh() {
    local candidates=()
    if [[ -n "${CONDA_EXE:-}" ]]; then
        candidates+=("$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh")
    fi
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        candidates+=("${CONDA_PREFIX}/etc/profile.d/conda.sh")
    fi
    if command -v conda >/dev/null 2>&1; then
        candidates+=("$(conda info --base)/etc/profile.d/conda.sh")
    fi
    candidates+=(
        "${HOME}/miniconda3/etc/profile.d/conda.sh"
        "${HOME}/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

install_miniconda() {
    local installer="/tmp/miniconda.sh"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${MINICONDA_URL}" -o "${installer}"
    else
        wget -q "${MINICONDA_URL}" -O "${installer}"
    fi

    bash "${installer}" -b -p "${MINICONDA_HOME}"
}

ensure_conda() {
    local conda_sh
    if conda_sh="$(find_conda_sh)"; then
        CONDA_SH_PATH="${conda_sh}"
        # shellcheck source=/dev/null
        source "${conda_sh}"
        return 0
    fi

    install_miniconda
    CONDA_SH_PATH="${MINICONDA_HOME}/etc/profile.d/conda.sh"
    # shellcheck source=/dev/null
    source "${CONDA_SH_PATH}"
}

persist_conda_shell_init() {
    local bashrc="${HOME}/.bashrc"
    local init_line="source ${CONDA_SH_PATH}"
    if [[ ! -f "${bashrc}" ]] || ! grep -qxF "${init_line}" "${bashrc}"; then
        printf '%s\n' "${init_line}" >> "${bashrc}"
    fi
}

conda_env_exists() {
    local env_name="$1"
    conda env list | awk -v name="${env_name}" '$1 == name {found=1} END {exit found ? 0 : 1}'
}

ensure_latency_env() {
    if conda_env_exists latency; then
        echo "[install] reusing existing conda env: latency" >&2
    else
        conda create -n latency -c conda-forge --override-channels python=3.10 -y
    fi

    conda activate latency
}

SELECTED_TORCH_PROFILE="$(resolve_torch_profile "${LATENCY_BENCH_TORCH_PROFILE}")"
export STARVLA_TORCH_PROFILE="${SELECTED_TORCH_PROFILE}"
# Speed-test mirrors once and route torch + pypi through the fastest provider. Exports
# STARVLA_TORCH_INDEX_BASE / PIP_INDEX_URL / UV_* so the requirements install and the
# starVLA bootstrap (and its parallel env builds) inherit the choice without re-probing.
export_mirror_env "${SELECTED_TORCH_PROFILE}"
TORCH_INDEX_URL="${STARVLA_TORCH_INDEX_BASE}/${SELECTED_TORCH_PROFILE}"
read -r -a TORCH_SPECS <<< "$(resolve_torch_specs "${SELECTED_TORCH_PROFILE}")"
echo "[install] torch_profile=${SELECTED_TORCH_PROFILE} index_url=${TORCH_INDEX_URL}"

# Fresh cloud instance bootstrap for the master branch.
#
# GitHub authentication on Vast.ai or similar rented GPU hosts:
#   1. On your local machine, make sure your GitHub SSH key is loaded:
#        ssh-add ~/.ssh/id_ed25519
#   2. SSH into the rented host with agent forwarding:
#        ssh -A -p <port> root@<host>
#   3. On the rented host, verify GitHub can see the forwarded agent:
#        ssh -T git@github.com
#
# Do not copy private SSH keys onto rented machines. If agent forwarding is not
# acceptable, use a GitHub deploy key or a short-lived token instead.
#

ensure_conda
persist_conda_shell_init

git submodule update --init --recursive

ensure_latency_env
# Keep uv's cache on the same filesystem as the conda envs so packages are
# hardlinked into each env (download/disk dedup) instead of silently copied.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(conda info --base)/.uv_cache}"
# Vast/CDN package downloads can stall long enough to exceed uv's default 30s
# HTTP timeout, especially for NVIDIA CUDA wheels.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_CONNECT_TIMEOUT="${UV_HTTP_CONNECT_TIMEOUT:-30}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
# Some cloud images (e.g. vast.ai) export UV_NO_CACHE=1 (disables uv's wheel cache, so
# every env re-downloads the identical multi-GB torch/CUDA stack) and UV_LINK_MODE=copy
# (defeats the cache->env hardlink dedup). Clear both to restore download/disk dedup.
unset UV_NO_CACHE UV_LINK_MODE
python -m pip install --upgrade pip
ensure_uv
pip_install "${TORCH_SPECS[@]}" --index-url "${TORCH_INDEX_URL}"
python - <<'PY'
import torch

print(f"[install] torch={torch.__version__} cuda={torch.version.cuda}")
PY
pip_install -r requirements.txt

bash scripts/bash_scripts/build_cuda_extensions.sh
bash starVLA/examples/rl_games/install/bootstrap.sh --split-envs
wandb login
