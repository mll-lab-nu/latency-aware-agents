#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-latency-openvla}"
PYTHON_VERSION="3.10"
REQUIREMENTS_FILE="${REPO_ROOT}/requirements_openvla.txt"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Missing requirements file: ${REQUIREMENTS_FILE}" >&2
    exit 1
fi

if [[ ! -d "${REPO_ROOT}/RL-inf" ]]; then
    echo "Missing RL-inf checkout at ${REPO_ROOT}/RL-inf" >&2
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
cd "${REPO_ROOT}"

python -m pip install --upgrade pip
python -m pip install -r "${REQUIREMENTS_FILE}"

# openvla-oft provides prismatic, but its metadata pins an old torch stack.
python -m pip install "git+https://github.com/moojink/openvla-oft.git" --no-deps --no-build-isolation

python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("prismatic")
if spec is None or spec.origin is None:
    raise SystemExit("prismatic package not found after openvla-oft install")

root = Path(spec.origin).parent
patch_text = '"""Minimal init patched for latency_bench OpenVLA eval."""\n'
for relative_path in ("__init__.py", "vla/__init__.py"):
    path = root / relative_path
    if not path.is_file():
        raise SystemExit(f"Expected prismatic file not found: {path}")
    path.write_text(patch_text, encoding="utf-8")
PY

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/RL-inf:${PYTHONPATH:-}"

python -c "from rlinf.models.embodiment.openvla_oft.rlinf import get_model; import latency_bench.policy.flappy_openvla; print('ok')"
