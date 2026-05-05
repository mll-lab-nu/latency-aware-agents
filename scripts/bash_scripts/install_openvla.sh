#!/usr/bin/env bash
set -euo pipefail

# Build a conda-based RLinf/OpenVLA-OFT environment for latency_bench Flappy eval.
#
# This uses RL-inf's OpenVLA-OFT implementation because latency_bench imports:
#   rlinf.models.embodiment.openvla_oft.rlinf.get_model
#
# The cloned ./openvla repository is intentionally not installed here; installing
# it can replace the prismatic package needed by RLinf's OpenVLA-OFT path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RLINF_DIR="${RLINF_DIR:-${REPO_ROOT}/RL-inf}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-latency-openvla}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
OPENVLA_OFT_SPEC="${OPENVLA_OFT_SPEC:-git+https://github.com/moojink/openvla-oft.git}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
FLASH_ATTN_SPEC="${FLASH_ATTN_SPEC:-flash-attn==2.7.4.post1}"

if [[ ! -d "${RLINF_DIR}" ]]; then
    echo "RLINF_DIR does not exist: ${RLINF_DIR}" >&2
    echo "Clone RL-inf under the repo root or set RLINF_DIR=/path/to/RL-inf." >&2
    exit 1
fi

if [[ ! -f "${RLINF_DIR}/pyproject.toml" ]]; then
    echo "RLinf pyproject.toml not found: ${RLINF_DIR}/pyproject.toml" >&2
    exit 1
fi

if [[ ! -d "${REPO_ROOT}/flappy-bird-gymnasium" ]]; then
    echo "Missing flappy-bird-gymnasium submodule at ${REPO_ROOT}/flappy-bird-gymnasium" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    echo "conda is required but was not found on PATH." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${CONDA_ENV_NAME}"

echo "[openvla] repo root: ${REPO_ROOT}"
echo "[openvla] RLinf dir: ${RLINF_DIR}"
echo "[openvla] conda env: ${CONDA_ENV_NAME}"
echo "[openvla] python: $(python --version)"

python -m pip install --upgrade pip setuptools wheel
# shellcheck disable=SC2086
python -m pip install ${TORCH_SPEC} --index-url "${TORCH_INDEX_URL}"
python -m pip install -e "${RLINF_DIR}[embodied]"

if [[ -n "${OPENVLA_OFT_DIR:-}" ]]; then
    if [[ ! -d "${OPENVLA_OFT_DIR}" ]]; then
        echo "OPENVLA_OFT_DIR does not exist: ${OPENVLA_OFT_DIR}" >&2
        exit 1
    fi
    python -m pip install -e "${OPENVLA_OFT_DIR}" --no-build-isolation
else
    python -m pip install "${OPENVLA_OFT_SPEC}" --no-build-isolation
fi

python -m pip install -e "${REPO_ROOT}/flappy-bird-gymnasium"
python -m pip install PyYAML==6.0.3 pillow==12.1.1

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
    python -m pip install "${FLASH_ATTN_SPEC}" --no-build-isolation
fi

activate_dir="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${activate_dir}"
cat > "${activate_dir}/latency_bench_openvla.sh" <<EOF
export LATENCY_BENCH_REPO_ROOT="${REPO_ROOT}"
export RLINF_DIR="${RLINF_DIR}"
export PYTHONPATH="${REPO_ROOT}:${RLINF_DIR}:\${PYTHONPATH:-}"
EOF

export LATENCY_BENCH_REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${RLINF_DIR}:${PYTHONPATH:-}"

python - <<'PY'
from rlinf.models.embodiment.openvla_oft.rlinf import get_model
import latency_bench.policy.flappy_openvla

print("OpenVLA/RLinf preflight ok:", get_model.__module__)
PY

cat <<EOF

OpenVLA conda environment is ready.

Activate it with:
  conda activate "${CONDA_ENV_NAME}"

Before full eval, set configs/experiments/flappy/openvla_sft_eval.yaml paths:
  policy.base_model_path
  policy.checkpoint_path

Keep evaluation.eval_parallel_envs=1 for flappy_openvla until predict_batch() exists.
EOF
