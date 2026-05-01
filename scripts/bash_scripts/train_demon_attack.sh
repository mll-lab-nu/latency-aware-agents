#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/bash_scripts/demon_attack.sh <latency>" >&2
    echo "Example: bash scripts/bash_scripts/demon_attack.sh 0" >&2
    exit 2
fi

LATENCY="$1"

if [[ ! "${LATENCY}" =~ ^[0-9]+$ ]]; then
    echo "Error: latency must be a non-negative integer, got '${LATENCY}'." >&2
    exit 2
fi

CONFIG_DIR="configs/train/demon_attack/latency_${LATENCY}/seed_0"

python -m training.train --config "${CONFIG_DIR}/train.yaml"
python -m training.train --config "${CONFIG_DIR}/eval.yaml"
python -m latency_bench.run --config "${CONFIG_DIR}/latency_bench_eval.yaml"
