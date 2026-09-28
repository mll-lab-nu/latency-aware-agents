#!/usr/bin/env bash
set -euo pipefail
# Usage: profile.sh <qwenoft|qwengr00t|qwenpi_v3> [BUNDLE] [--publish]
# CPU environment/rendering; expose only the RTX 3090 used for inference.
MODEL="$1"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
args=(run --model "${MODEL}" --workload atlantis)
if [[ $# -gt 0 && "$1" != --* ]]; then
  args+=(--model-path "$1")
  shift
fi
exec bash scripts/profile.sh "${args[@]}" "$@"
