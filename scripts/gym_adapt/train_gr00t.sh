#!/usr/bin/env bash
set -euo pipefail

# The caller selects the visible training GPUs.
: "${CUDA_VISIBLE_DEVICES:?Reserve explicit idle GPUs before launch}"
TRAIN_CONFIG=$(realpath "$1")
NUM_PROCESSES=$2
MAIN_PROCESS_PORT=$3
PYTHON_BIN=$4
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/starVLA:$REPO_ROOT/third_party/sample-factory${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT/third_party/starVLA"
exec "$PYTHON_BIN" -m accelerate.commands.launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "$NUM_PROCESSES" --main_process_port "$MAIN_PROCESS_PORT" \
  starVLA/training/train_starvla.py --config_yaml "$TRAIN_CONFIG"
