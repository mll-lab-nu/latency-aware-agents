#!/usr/bin/env bash
set -euo pipefail

export LATENCY_BENCH_ENV_BACKEND_START_METHOD=spawn
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
python -m latency_bench.cross_device.protocol "$@"
