#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
exec python -m latency_bench.cross_device.study "$@"
