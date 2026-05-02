#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/bash_scripts/eval_flappy.sh" >&2
    echo "Error: this script does not take a latency argument; it scans all generated flappy eval configs." >&2
    exit 2
fi

CONFIG_ROOT="configs/train/flappy"

mapfile -t CONFIG_DIRS < <(
    find "${CONFIG_ROOT}" -maxdepth 3 -mindepth 3 -type f -path "${CONFIG_ROOT}/latency_*/seed_0/eval.yaml" \
        -printf '%h\n' | sort -V
)

if [[ ${#CONFIG_DIRS[@]} -eq 0 ]]; then
    echo "Error: no eval configs found under ${CONFIG_ROOT}/latency_*/seed_0." >&2
    exit 2
fi

for CONFIG_DIR in "${CONFIG_DIRS[@]}"; do
    echo "==> Evaluating ${CONFIG_DIR}"
    python -m training.train --config "${CONFIG_DIR}/eval.yaml"
    python -m latency_bench.run --config "${CONFIG_DIR}/latency_bench_eval.yaml"
done
