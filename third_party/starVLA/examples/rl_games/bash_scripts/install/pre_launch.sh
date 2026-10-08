#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../install/_host.sh"
export PYTHONPATH="${LATENCY_BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
