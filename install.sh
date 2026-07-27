#!/usr/bin/env bash
set -euo pipefail

INSTALL_TIER="dev"
MODEL_TARGET=""
ACCEPT_ROM_LICENSE="false"
MINICONDA_HOME="${HOME}/miniconda3"
MINICONDA_VERSION="py310_26.1.1-1"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh"
CONDA_SH_PATH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: bash install.sh [options]

Installs one or all isolated RL-games model environments with:
Python 3.10, CUDA Toolkit 12.8.1, and PyTorch 2.11.0+cu128.

Options:
  --tier <use|dev>         Install runtime or development dependencies (default: dev)
  --model <name|all>       Required: openvla|pi0|pi05|gr00t|wan_oft|all
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
    openvla|pi0|pi05|gr00t|wan_oft|all) ;;
    "")
        echo "[install] --model is required" >&2
        usage >&2
        exit 1
        ;;
    *)
        echo "[install] invalid model '${MODEL_TARGET}'; expected openvla|pi0|pi05|gr00t|wan_oft|all" >&2
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

ensure_conda
persist_conda_shell_init

cd "${SCRIPT_DIR}"
git submodule update --init --recursive

BOOTSTRAP_ARGS=(
    --tier "${INSTALL_TIER}"
    --model "${MODEL_TARGET}"
)
if [[ "${ACCEPT_ROM_LICENSE}" == "true" ]]; then
    BOOTSTRAP_ARGS+=(--accept-rom-license)
fi

echo "[install] fixed stack: python=3.10 cuda=12.8.1 torch=2.11.0+cu128 model=${MODEL_TARGET}"
bash third_party/starVLA/examples/rl_games/install/bootstrap.sh "${BOOTSTRAP_ARGS[@]}"
