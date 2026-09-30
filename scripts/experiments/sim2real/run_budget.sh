#!/usr/bin/env bash
# One profiling-host campaign. Each stage completes on both GPUs before advancing.
set -euo pipefail
ROOT="${1:?usage: run_budget.sh OUTPUT_ROOT [PYTHON]}"
PYTHON="${2:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs"
for stage in verify real fit; do
    echo "START stage=$stage"
    "$PYTHON" -m latency_bench.cross_device.budget --stage "$stage" --model qwenoft --root "$ROOT" >"$ROOT/logs/${stage}_qwenoft.log" 2>&1 &
    oft_pid=$!
    "$PYTHON" -m latency_bench.cross_device.budget --stage "$stage" --model qwengr00t --root "$ROOT" >"$ROOT/logs/${stage}_qwengr00t.log" 2>&1 &
    groot_pid=$!
    status=0
    wait "$oft_pid" || status=$?
    wait "$groot_pid" || status=$?
    if (( status != 0 )); then
        echo "FAILED stage=$stage; see $ROOT/logs" >&2
        exit "$status"
    fi
    echo "COMPLETE stage=$stage"
done
echo "COMPLETE real/profile inputs; retirement controller owns upload verification and shutdown"
