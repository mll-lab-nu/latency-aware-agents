#!/usr/bin/env bash
set -euo pipefail

INSTALL_TIER="use"
MODEL_TARGET="openvla"
TORCH_PROFILE="${LATENCY_BENCH_TORCH_PROFILE:-auto}"
ACCEPT_ROM_LICENSE="false"
REQUESTED_ENVS=()
MINICONDA_HOME="${HOME}/miniconda3"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
CONDA_SH_PATH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/bash_scripts/_pip.sh
source "${SCRIPT_DIR}/scripts/bash_scripts/_pip.sh"
# shellcheck source=scripts/bash_scripts/_mirrors.sh
source "${SCRIPT_DIR}/scripts/bash_scripts/_mirrors.sh"

usage() {
    cat <<'EOF'
Usage: bash install.sh [options]

Options:
  --tier <use|dev>         Install runtime or development dependencies (default: use)
  --model <name>           openvla|pi0|pi05|gr00t|wan_oft (default: openvla)
  --env <name|all>         Repeatable: flappy|demon_attack|deadly_corridor|all (default: all)
  --torch-profile <name>   auto|cpu|cu126|cu128|cu130 (default: auto)
  --accept-rom-license     Permit AutoROM to accept and download Atari ROMs
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            INSTALL_TIER="$2"
            shift 2
            ;;
        --model)
            MODEL_TARGET="$2"
            shift 2
            ;;
        --env)
            REQUESTED_ENVS+=("$2")
            shift 2
            ;;
        --torch-profile)
            TORCH_PROFILE="$2"
            shift 2
            ;;
        --accept-rom-license)
            ACCEPT_ROM_LICENSE="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[install] unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "${INSTALL_TIER}" in
    use|dev) ;;
    *)
        echo "[install] invalid tier '${INSTALL_TIER}'; expected use|dev" >&2
        exit 1
        ;;
esac

case "${MODEL_TARGET}" in
    openvla|pi0|pi05|gr00t|wan_oft) ;;
    *)
        echo "[install] invalid model '${MODEL_TARGET}'; expected openvla|pi0|pi05|gr00t|wan_oft" >&2
        exit 1
        ;;
esac

ENVS_TO_INSTALL=()
STARVLA_ENV_ARGS=()
if [[ ${#REQUESTED_ENVS[@]} -eq 0 ]]; then
    ENVS_TO_INSTALL=(flappy demon_attack deadly_corridor)
    STARVLA_ENV_ARGS=(--env all)
else
    if [[ ${#REQUESTED_ENVS[@]} -gt 1 ]]; then
        for env_name in "${REQUESTED_ENVS[@]}"; do
            if [[ "${env_name}" == "all" ]]; then
                echo "[install] --env all cannot be combined with another --env" >&2
                exit 1
            fi
        done
    fi
    for env_name in "${REQUESTED_ENVS[@]}"; do
        case "${env_name}" in
            all)
                ENVS_TO_INSTALL=(flappy demon_attack deadly_corridor)
                ;;
            flappy|demon_attack|deadly_corridor)
                ENVS_TO_INSTALL+=("${env_name}")
                ;;
            *)
                echo "[install] invalid env '${env_name}'; expected flappy|demon_attack|deadly_corridor|all" >&2
                exit 1
                ;;
        esac
        STARVLA_ENV_ARGS+=(--env "${env_name}")
    done
fi

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

cd "${SCRIPT_DIR}"
git submodule update --init --recursive

# Keep root and model environments on the same Torch profile implementation.
# shellcheck source=third_party/starVLA/examples/rl_games/install/_torch_profile.sh
source "${SCRIPT_DIR}/third_party/starVLA/examples/rl_games/install/_torch_profile.sh"
SELECTED_TORCH_PROFILE="$(resolve_torch_profile "${TORCH_PROFILE}")"
export STARVLA_TORCH_PROFILE="${SELECTED_TORCH_PROFILE}"
export ACCEPT_ROM_LICENSE

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

# Speed-test mirrors once and share the selected provider with every model env.
export_mirror_env "${SELECTED_TORCH_PROFILE}"
echo "[install] tier=${INSTALL_TIER} model=${MODEL_TARGET} torch=${SELECTED_TORCH_PROFILE} envs=${ENVS_TO_INSTALL[*]}"

python -m pip install --upgrade pip
ensure_uv
PYTHON_BIN=python bash third_party/starVLA/examples/rl_games/install/torch.sh

if [[ "${INSTALL_TIER}" == "dev" ]]; then
    pip_install -e ".[dev]"
else
    pip_install -e .
fi
pip_install --no-deps -e third_party/flappy-bird-gymnasium
pip_install -e third_party/sample-factory

for env_name in "${ENVS_TO_INSTALL[@]}"; do
    PYTHON_BIN=python bash "third_party/starVLA/examples/rl_games/install/env/${env_name}.sh"
done

if [[ "${SELECTED_TORCH_PROFILE}" != "cpu" ]]; then
    pip_install ninja==1.13.0
    bash scripts/bash_scripts/build_cuda_extensions.sh
fi

BOOTSTRAP_ARGS=(
    --tier "${INSTALL_TIER}"
    --model "${MODEL_TARGET}"
    --torch-profile "${SELECTED_TORCH_PROFILE}"
)
BOOTSTRAP_ARGS+=("${STARVLA_ENV_ARGS[@]}")
if [[ "${ACCEPT_ROM_LICENSE}" == "true" ]]; then
    BOOTSTRAP_ARGS+=(--accept-rom-license)
fi
bash third_party/starVLA/examples/rl_games/install/bootstrap.sh "${BOOTSTRAP_ARGS[@]}"
