#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
# shellcheck source=../_pip.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_pip.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_host.sh"

pip_install -e "${LATENCY_BENCH_ROOT}/.[atari]"
if ! "$PYTHON_BIN" -m scripts.install.validate atari >/dev/null 2>&1; then
  AutoROM --accept-license
fi
"$PYTHON_BIN" -m scripts.install.validate atari

echo "[install/env/demon_attack] done"
