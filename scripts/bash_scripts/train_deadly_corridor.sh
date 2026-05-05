#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/bash_scripts/train_deadly_corridor.sh <latency-or-distribution-label>" >&2
    echo "Example: bash scripts/bash_scripts/train_deadly_corridor.sh 0" >&2
    echo "Example: bash scripts/bash_scripts/train_deadly_corridor.sh normal_mean_5_std_2_min_0_max_15" >&2
    exit 2
fi

LATENCY_SPEC="$1"

if [[ ! "${LATENCY_SPEC}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "Error: latency spec must be a non-negative integer or canonical distribution label, got '${LATENCY_SPEC}'." >&2
    exit 2
fi

if [[ "${LATENCY_SPEC}" == latency_* ]]; then
    LATENCY_DIR="${LATENCY_SPEC}"
else
    LATENCY_DIR="latency_${LATENCY_SPEC}"
fi

CONFIG_DIR="configs/train/deadly_corridor/${LATENCY_DIR}/seed_0"

python -m training.train --config "${CONFIG_DIR}/train.yaml"
python -m training.train --config "${CONFIG_DIR}/eval.yaml"
python -m latency_bench.run --config "${CONFIG_DIR}/latency_bench_eval.yaml"
