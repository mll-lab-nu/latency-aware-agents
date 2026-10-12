#!/usr/bin/env bash
# Set the teacher root in the YAML; optional arguments are collect CLI flags.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m latency_bench collect --config configs/examples/air_raid/collect.yaml "$@"
