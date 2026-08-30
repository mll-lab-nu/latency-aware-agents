#!/usr/bin/env bash
set -euo pipefail

INSTALL_TIER="dev"
MODEL_TARGET=""
SKIP_FLASH_ATTN="false"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.8.1"
LATENCY_ENV_NAME="latency"
MINICONDA_HOME="${HOME}/miniconda3"
MINICONDA_VERSION="py310_26.1.1-1"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh"
CONDA_SH_PATH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: bash install.sh [options]

Installs the shared latency environment and one or all isolated RL-games model
environments with:
Python 3.10, CUDA Toolkit 12.8.1, and PyTorch 2.11.0+cu128.

Options:
  --tier <use|dev>         Install runtime or development dependencies (default: dev)
  --model <name|all>       Required: openvla|pi0|pi05|gr00t|official_gr00t|wan_oft|all
  --skip-flash-attn        Skip FlashAttention install/check in selected model envs
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
        --skip-flash-attn)
            SKIP_FLASH_ATTN="true"
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
    openvla|pi0|pi05|gr00t|official_gr00t|wan_oft|all) ;;
    "")
        echo "[install] --model is required" >&2
        usage >&2
        exit 1
        ;;
    *)
        echo "[install] invalid model '${MODEL_TARGET}'; expected openvla|pi0|pi05|gr00t|official_gr00t|wan_oft|all" >&2
        exit 1
        ;;
esac

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
    local installer="/tmp/miniconda-${MINICONDA_VERSION}.sh"
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
        return
    fi

    install_miniconda
    CONDA_SH_PATH="${MINICONDA_HOME}/etc/profile.d/conda.sh"
    # shellcheck source=/dev/null
    source "${CONDA_SH_PATH}"
}

persist_conda_shell_init() {
    local init_line="source ${CONDA_SH_PATH}"
    if [[ ! -f "${HOME}/.bashrc" ]] || ! grep -qxF "${init_line}" "${HOME}/.bashrc"; then
        printf '%s\n' "${init_line}" >> "${HOME}/.bashrc"
    fi
}

ensure_latency_env() {
    local conda_action="create"
    local -a conda_packages=(
        "python=${PYTHON_VERSION}"
        "cuda-toolkit=${CUDA_VERSION}"
        pip
        setuptools
        wheel
        packaging
        "zlib=1.3.2"
    )
    if conda env list | awk '{print $1}' | grep -qx "${LATENCY_ENV_NAME}"; then
        conda_action="install"
        echo "[install] updating existing Conda environment: ${LATENCY_ENV_NAME}"
    else
        echo "[install] creating Conda environment: ${LATENCY_ENV_NAME}"
    fi

    if [[ "${MODEL_TARGET}" == "official_gr00t" || "${MODEL_TARGET}" == "all" ]]; then
        conda_packages+=("ffmpeg=7")
    fi

    conda "${conda_action}" \
        -n "${LATENCY_ENV_NAME}" \
        --override-channels \
        -c nvidia/label/cuda-12.8.1 \
        -c conda-forge \
        "${conda_packages[@]}" \
        -y
}

install_latency_env() {
    local original_ld_library_path="${LD_LIBRARY_PATH:-}"

    ensure_latency_env
    set +u
    conda activate "${LATENCY_ENV_NAME}"
    set -u

    export CUDA_HOME="${CONDA_PREFIX}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${original_ld_library_path:+:${original_ld_library_path}}"
    export TORCH_CUDA_ARCH_LIST="9.0;12.0"

    PYTHON_BIN=python
    export PYTHON_BIN
    # shellcheck source=third_party/starVLA/examples/rl_games/install/_pip.sh
    source third_party/starVLA/examples/rl_games/install/_pip.sh

    python -m pip install pip==25.3 setuptools==80.9.0 wheel==0.47.0
    ensure_uv
    PYTHON_BIN=python bash third_party/starVLA/examples/rl_games/install/torch.sh
    local root_package=".[dev]"
    if [[ "${MODEL_TARGET}" == "official_gr00t" || "${MODEL_TARGET}" == "all" ]]; then
        root_package=".[dev,sonic]"
    fi
    pip_install -e "${root_package}"
    if [[ "${MODEL_TARGET}" == "official_gr00t" || "${MODEL_TARGET}" == "all" ]]; then
        pip_install --no-deps -e third_party/humanoid-bench
        python -c 'import humanoid_bench, mujoco, onnxruntime; print("latency Sonic imports complete")'
    fi
    pip_install -e third_party/sample-factory

    for env_name in demon_attack deadly_corridor flappy; do
        PYTHON_BIN=python bash "third_party/starVLA/examples/rl_games/install/env/${env_name}.sh"
    done

    pip_install ninja==1.13.0
    bash scripts/bash_scripts/build_cuda_extensions.sh
    python -m pip check
    conda deactivate

    export LD_LIBRARY_PATH="${original_ld_library_path}"
    echo "[install] latency environment complete"
}

ensure_official_gr00t_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        wget -qO- https://astral.sh/uv/install.sh | sh
    fi
    export PATH="${HOME}/.local/bin:${PATH}"
}

install_official_gr00t() {
    local original_ld_library_path="${LD_LIBRARY_PATH:-}"
    set +u
    conda activate "${LATENCY_ENV_NAME}"
    set -u
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64${original_ld_library_path:+:${original_ld_library_path}}"
    ensure_official_gr00t_uv
    echo "[install] syncing Isaac-GR00T Python 3.12 environment"
    uv sync --project third_party/Isaac-GR00T --python 3.12
    uv run --project third_party/Isaac-GR00T python -c \
        'import gr00t, torchcodec; from gr00t.policy import Gr00tPolicy; print(Gr00tPolicy.__name__)'
    conda deactivate
    export LD_LIBRARY_PATH="${original_ld_library_path}"
}

ensure_conda
persist_conda_shell_init

cd "${SCRIPT_DIR}"
git submodule update --init --recursive

CONDA_BASE="$(conda info --base)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CONDA_BASE}/.uv_cache}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_CONNECT_TIMEOUT="${UV_HTTP_CONNECT_TIMEOUT:-30}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
unset UV_NO_CACHE UV_LINK_MODE

echo "[install] fixed stack: python=3.10 cuda=12.8.1 torch=2.11.0+cu128 model=${MODEL_TARGET}"
install_latency_env

if [[ "${MODEL_TARGET}" == "official_gr00t" || "${MODEL_TARGET}" == "all" ]]; then
    install_official_gr00t
fi

if [[ "${MODEL_TARGET}" == "official_gr00t" ]]; then
    exit 0
fi

BOOTSTRAP_ARGS=(
    --tier "${INSTALL_TIER}"
    --model "${MODEL_TARGET}"
)

if [[ "${SKIP_FLASH_ATTN}" == "true" ]]; then
    BOOTSTRAP_ARGS+=(--skip-flash-attn)
fi

bash third_party/starVLA/examples/rl_games/install/bootstrap.sh "${BOOTSTRAP_ARGS[@]}"
