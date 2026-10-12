#!/usr/bin/env bash
# Supply --config and any supported collect CLI overrides.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON:-python}" -m latency_bench collect "$@"
