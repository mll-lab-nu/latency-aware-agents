#!/usr/bin/env bash
set -euo pipefail

TORCH_VERSION="2.10.0"
LATENCY_BENCH_TORCH_PROFILE="${LATENCY_BENCH_TORCH_PROFILE:-auto}"
MINICONDA_HOME="${HOME}/miniconda3"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

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

torch_index_url() {
    case "$1" in
        cu126)
            echo "https://download.pytorch.org/whl/cu126"
            ;;
        cu128)
            echo "https://download.pytorch.org/whl/cu128"
            ;;
        cu130)
            echo "https://download.pytorch.org/whl/cu130"
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
        # shellcheck source=/dev/null
        source "${conda_sh}"
        return 0
    fi

    install_miniconda
    # shellcheck source=/dev/null
    source "${MINICONDA_HOME}/etc/profile.d/conda.sh"
}

ensure_envpool_build_tools() {
    export PATH="${HOME}/go/bin:${PATH}"
    export CGO_ENABLED=0
    export USE_BAZEL_VERSION=6.5.0
    if command -v go >/dev/null 2>&1; then
        echo "[install] using existing $(go version)"
        return
    fi

    echo "[install] installing go from conda-forge for envpool build"
    set +u
    conda install -y -c conda-forge go
    set -u
}

SELECTED_TORCH_PROFILE="$(resolve_torch_profile "${LATENCY_BENCH_TORCH_PROFILE}")"
TORCH_INDEX_URL="$(torch_index_url "${SELECTED_TORCH_PROFILE}")"
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

git submodule update --init --recursive

conda create -n latency python=3.10 -y

conda activate latency
ensure_envpool_build_tools
python -m pip install --upgrade pip
python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
python - <<'PY'
import torch

print(f"[install] torch={torch.__version__} cuda={torch.version.cuda}")
PY
python -m pip install -r requirements.txt

# install envpool for deadly corridor acceleration
rm -f envpool/dist/envpool-*.whl
make -C envpool bazel-release
python -m pip install --force-reinstall --no-deps envpool/dist/envpool-*.whl
bash scripts/bash_scripts/build_cuda_extensions.sh
wandb login
