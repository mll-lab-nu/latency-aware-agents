#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# shellcheck source=_host.sh
source "${SCRIPT_DIR}/_host.sh"
# shellcheck source=_pip.sh
source "${SCRIPT_DIR}/_pip.sh"
source "${LATENCY_BENCH_ROOT}/scripts/install/_tasks.sh"

INSTALL_TIER="use"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.8.1"
MODEL_TARGET=""
ENV_TARGET=""
RUN_VALIDATE="true"
SKIP_FLASH_ATTN="false"
SUPPORTED_MODELS=(openvla pi0 pi05 gr00t wan_oft)

usage() {
  cat <<EOF
Usage: bash examples/rl_games/install/bootstrap.sh [options]

Installs one or all fixed RL-games model environments:
Python ${PYTHON_VERSION}, CUDA Toolkit ${CUDA_VERSION}, PyTorch 2.11.0+cu128.

Options:
  --tier <use|dev>         Dependency tier (default: ${INSTALL_TIER})
  --model <name|all>       Required: openvla|pi0|pi05|gr00t|wan_oft|all
  --env <tasks|all>        Install selected tasks for in-process training evaluation
  --skip-validate          Skip final validation
  --skip-flash-attn        Skip FlashAttention install/check for model envs
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
      ENV_TARGET="$2"
      shift 2
      ;;
    --skip-validate)
      RUN_VALIDATE="false"
      shift
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
      echo "[bootstrap] unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${INSTALL_TIER}" in
  use|dev) ;;
  *)
    echo "[bootstrap] invalid tier '${INSTALL_TIER}'; expected use|dev" >&2
    exit 1
    ;;
esac

case "${MODEL_TARGET}" in
  openvla|pi0|pi05|gr00t|wan_oft)
    MODELS_TO_INSTALL=("${MODEL_TARGET}")
    ;;
  all)
    MODELS_TO_INSTALL=("${SUPPORTED_MODELS[@]}")
    ;;
  "")
    echo "[bootstrap] --model is required" >&2
    usage >&2
    exit 1
    ;;
  *)
    echo "[bootstrap] invalid model '${MODEL_TARGET}'; expected openvla|pi0|pi05|gr00t|wan_oft|all" >&2
    exit 1
    ;;
esac

resolve_install_tasks "${ENV_TARGET}"
export STARVLA_INSTALL_TIER="${INSTALL_TIER}"
export STARVLA_SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN}"

ensure_conda_env() {
  local env_name="$1"
  local conda_action="create"
  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    conda_action="install"
    echo "[bootstrap] updating existing Conda environment: ${env_name}"
  else
    echo "[bootstrap] creating Conda environment: ${env_name}"
  fi
  conda "${conda_action}" \
    -n "${env_name}" \
    --override-channels \
    -c nvidia/label/cuda-12.8.1 \
    -c conda-forge \
    "python=${PYTHON_VERSION}" \
    "cuda-toolkit=${CUDA_VERSION}" \
    pip \
    setuptools \
    wheel \
    packaging \
    "zlib=1.3.2" \
    "ffmpeg=7" \
    -y
}

run_validation() {
  local model="$1"

  PYTHON_BIN=python "${SCRIPT_DIR}/validate/common.sh"
  case "${model}" in
    openvla|pi0|pi05|gr00t)
      if [[ "${STARVLA_SKIP_FLASH_ATTN}" != "true" ]]; then
        PYTHON_BIN=python "${SCRIPT_DIR}/flash_attn.sh" --check
      else
        echo "[bootstrap] skipping flash-attn validation"
      fi
      ;;
  esac

}

command -v conda >/dev/null
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CONDA_BASE}/.uv_cache}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_CONNECT_TIMEOUT="${UV_HTTP_CONNECT_TIMEOUT:-30}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
unset UV_NO_CACHE UV_LINK_MODE
ORIGINAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

for model in "${MODELS_TO_INSTALL[@]}"; do
  CONDA_ENV_NAME="${STARVLA_CONDA_ENV_NAME:-starvla_rl_games_${model}}"
  ensure_conda_env "${CONDA_ENV_NAME}"
  set +u
  conda activate "${CONDA_ENV_NAME}"
  set -u

  export CUDA_HOME="${CONDA_PREFIX}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${ORIGINAL_LD_LIBRARY_PATH:+:${ORIGINAL_LD_LIBRARY_PATH}}"
  export TORCH_CUDA_ARCH_LIST="8.6;9.0;12.0"

  echo "[bootstrap] tier=${INSTALL_TIER} model=${model}"
  PYTHON_BIN=python "${SCRIPT_DIR}/common.sh"
  PYTHON_BIN=python "${SCRIPT_DIR}/model/${model}.sh"
  PYTHON_BIN=python
  export PYTHON_BIN
  install_selected_tasks

  # Dataset export dependencies are installed by the root project.
  python -m pip install --no-deps 'lerobot==0.4.4'

  if [[ "${RUN_VALIDATE}" == "true" ]]; then
    run_validation "${model}"
  fi

  echo "[bootstrap] model=${model} complete"
  echo "[bootstrap] activate with: conda activate ${CONDA_ENV_NAME}"
  conda deactivate
done

export LD_LIBRARY_PATH="${ORIGINAL_LD_LIBRARY_PATH}"
echo "[bootstrap] complete"
echo "[bootstrap] repo root: ${REPO_ROOT}"
