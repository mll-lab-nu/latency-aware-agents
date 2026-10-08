#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/pi0.sh"

echo "[install/model/pi05] tier=${STARVLA_INSTALL_TIER:-use} done"
