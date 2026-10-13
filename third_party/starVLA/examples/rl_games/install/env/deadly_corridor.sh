#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"

pip_install "vizdoom==1.3.0" "gymnasium==0.29.1"
"$PYTHON_BIN" -m scripts.install.validate deadly_corridor

echo "[install/env/deadly_corridor] done"
