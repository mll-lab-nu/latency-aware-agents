#!/usr/bin/env bash

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${INSTALL_DIR}/../../.." && pwd)"

LATENCY_BENCH_ROOT="${LATENCY_BENCH_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

export STARVLA_ROOT LATENCY_BENCH_ROOT
export PYTHONPATH="${LATENCY_BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
