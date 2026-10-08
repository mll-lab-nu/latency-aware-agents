#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INSTALL_TIER="${STARVLA_INSTALL_TIER:-use}"

# shellcheck source=_pip.sh
source "${SCRIPT_DIR}/_pip.sh"
# shellcheck source=_host.sh
source "${SCRIPT_DIR}/_host.sh"

"$PYTHON_BIN" - <<'PY'
import sys

assert sys.version_info[:2] == (3, 10), sys.version
PY

"$PYTHON_BIN" -m pip install pip==25.3 setuptools==80.9.0 wheel==0.47.0
ensure_uv
"${SCRIPT_DIR}/torch.sh"

PYTORCH3D_NO_EXTENSION=1 "$PYTHON_BIN" -m pip install \
    --force-reinstall \
    --no-deps \
    --no-build-isolation \
    --no-binary=pipablepytorch3d \
    pipablepytorch3d==0.7.6
pip_install -r "$REPO_ROOT/requirements.txt"
pip_install -e "$REPO_ROOT"
pip_install -e "$LATENCY_BENCH_ROOT"
pip_install --no-deps -e "$LATENCY_BENCH_ROOT/third_party/flappy-bird-gymnasium"
pip_install -e "$LATENCY_BENCH_ROOT/third_party/sample-factory"
"$PYTHON_BIN" -m pip install --force-reinstall --no-deps eva-decord==0.6.1

if [[ "${INSTALL_TIER}" == "dev" ]]; then
    pip_install -r "$REPO_ROOT/requirements-dev.txt"
    pip_install -e "$REPO_ROOT[dev]"
fi

echo "[install/common] tier=${INSTALL_TIER} done"
