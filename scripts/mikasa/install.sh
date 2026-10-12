#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${REPO_ROOT}/third_party/starVLA/examples/rl_games/install/_pip.sh"

pip_install --prerelease allow -e "${REPO_ROOT}/third_party/MIKASA-Robo"
