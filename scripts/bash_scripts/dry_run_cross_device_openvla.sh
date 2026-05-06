#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DRY_RUN=1
exec bash "${SCRIPT_DIR}/run_cross_device_openvla.sh"
