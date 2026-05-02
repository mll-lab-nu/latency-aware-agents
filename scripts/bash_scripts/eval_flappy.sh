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

DEFAULT_CONFIG_ROOT="configs/train/flappy"
CONFIG_ROOT="${LATENCY_BENCH_EVAL_CONFIG_ROOT:-${DEFAULT_CONFIG_ROOT}}"

read_config_value() {
    local config_path="$1"
    local dotted_key="$2"
    local section="${dotted_key%%.*}"
    local key="${dotted_key#*.}"
    awk -v section="${section}" -v key="${key}" '
        /^[^[:space:]][^:]*:/ {
            current = $1
            sub(":$", "", current)
            next
        }
        current == section {
            line = $0
            sub(/^[[:space:]]+/, "", line)
            if (line ~ "^" key ":") {
                sub(/^[^:]*:[[:space:]]*/, "", line)
                sub(/[[:space:]]+#.*$/, "", line)
                print line
                exit
            }
        }
    ' "${config_path}"
}

mapfile -t CONFIG_DIRS < <(
    find "${CONFIG_ROOT}" -maxdepth 3 -mindepth 3 -type f -path "${CONFIG_ROOT}/latency_*/seed_0/eval.yaml" \
        -printf '%h\n' | sort -V
)

if [[ ${#CONFIG_DIRS[@]} -eq 0 ]]; then
    echo "Error: no eval configs found under ${CONFIG_ROOT}/latency_*/seed_0." >&2
    exit 2
fi

EVAL_RUNS=0
EVAL_FAILURES=0

for CONFIG_DIR in "${CONFIG_DIRS[@]}"; do
    EXPERIMENT_NAME="$(read_config_value "${CONFIG_DIR}/eval.yaml" "experiment.name")"
    TRAIN_DIR="$(read_config_value "${CONFIG_DIR}/eval.yaml" "backend.train_dir")"
    CHECKPOINT_CONFIG="${TRAIN_DIR}/${EXPERIMENT_NAME}/config.json"
    if [[ ! -f "${CHECKPOINT_CONFIG}" ]]; then
        echo "Skipping ${CONFIG_DIR}: missing checkpoint config ${CHECKPOINT_CONFIG}" >&2
        continue
    fi

    echo "==> Evaluating ${CONFIG_DIR}"
    EVAL_RUNS=$((EVAL_RUNS + 1))
    if ! python -m training.train --config "${CONFIG_DIR}/eval.yaml"; then
        echo "Eval failed for ${CONFIG_DIR}: training eval command failed" >&2
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
        continue
    fi
    if ! python -m latency_bench.run --config "${CONFIG_DIR}/latency_bench_eval.yaml"; then
        echo "Eval failed for ${CONFIG_DIR}: latency_bench eval command failed" >&2
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
        continue
    fi
done

if [[ ${EVAL_RUNS} -eq 0 ]]; then
    echo "Error: no checkpoint-backed eval configs found under ${CONFIG_ROOT}." >&2
    exit 2
fi

if [[ ${EVAL_FAILURES} -ne 0 ]]; then
    echo "Error: ${EVAL_FAILURES} eval run(s) failed; completed remaining checkpoint-backed configs." >&2
    exit 1
fi
