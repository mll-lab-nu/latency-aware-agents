#!/usr/bin/env bash
# Collect successful H1 episodes, convert them, and save local provenance.
set -euo pipefail
PACKAGE_ONLY=0
if [[ "${1:-}" == --package-only ]]; then
  PACKAGE_ONLY=1
  shift
fi
if [[ $# -ne 4 ]]; then
  echo "Usage: $0 [--package-only] <latency-frames> <gpu-id> <checkpoint> <output-root>" >&2
  exit 2
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
TASK_PY="${TASK_PY:?simulator interpreter}"
LEROBOT_PY="${LEROBOT_PY:?installed LeRobot interpreter}"
export CUDA_VISIBLE_DEVICES="$2"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/MIKASA-Robo"
OUTPUT="$4/fixed_l$1"
CONFIG=configs/examples/mikasa_intercept_grab_fast/collect_successful_h1.yaml
ARGS=(--config "$CONFIG" --latency-frames "$1" --checkpoint "$3" --output "$OUTPUT")
PREPARE_ARGS=()
if [[ "$PACKAGE_ONLY" == 1 ]]; then PREPARE_ARGS+=(--package-only); fi

"$TASK_PY" -m scripts.tasks.mikasa_intercept_grab_fast.successful_h1_dataset --stage prepare "${ARGS[@]}" "${PREPARE_ARGS[@]}"
if [[ "$PACKAGE_ONLY" == 0 ]]; then
  "$TASK_PY" -m latency_bench collect --config "$OUTPUT/collect.yaml"
fi
"$LEROBOT_PY" -m latency_bench convert --config "$OUTPUT/convert.yaml"
"$LEROBOT_PY" -m scripts.tasks.mikasa_intercept_grab_fast.successful_h1_dataset --stage finalize "${ARGS[@]}"
