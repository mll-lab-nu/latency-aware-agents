#!/usr/bin/env bash

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${INSTALL_DIR}/../../.." && pwd)"

if [[ -z "${LATENCY_BENCH_ROOT:-}" ]]; then
  LATENCY_BENCH_ROOT="$(git -C "${STARVLA_ROOT}" rev-parse --show-superproject-working-tree 2>/dev/null || true)"
fi

if [[ -z "${LATENCY_BENCH_ROOT:-}" ]]; then
  # starVLA isn't checked out as a git submodule (e.g. plain sibling clones
  # under a shared workspace dir); fall back to the sibling directory.
  LATENCY_BENCH_ROOT="$(dirname "${STARVLA_ROOT}")/latency-sensitive-bench"
fi

if [[ ! -d "${LATENCY_BENCH_ROOT}" ]]; then
  echo "[_host] LATENCY_BENCH_ROOT could not be resolved (tried: '${LATENCY_BENCH_ROOT}')." >&2
  echo "[_host] Set LATENCY_BENCH_ROOT explicitly to the latency-sensitive-bench checkout." >&2
  exit 1
fi

export STARVLA_ROOT LATENCY_BENCH_ROOT
