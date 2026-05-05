#!/usr/bin/env bash

set -euo pipefail

ENVS="${ENVS:-flappy,demon_attack,deadly_corridor}"
SEEDS="${SEEDS:-0}"
FLAPPY_LATENCIES="${FLAPPY_LATENCIES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
DEMON_ATTACK_LATENCIES="${DEMON_ATTACK_LATENCIES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
DEADLY_CORRIDOR_LATENCIES="${DEADLY_CORRIDOR_LATENCIES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
GENERATED_ROOT="${GENERATED_ROOT:-configs/train}"
TRAIN_DIR="${TRAIN_DIR:-results/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
DEVICE="${DEVICE:-gpu}"

python scripts/generate_sample_factory_same_latency_sweep.py \
  --envs "${ENVS}" \
  --seeds "${SEEDS}" \
  --flappy-latencies "${FLAPPY_LATENCIES}" \
  --demon-attack-latencies "${DEMON_ATTACK_LATENCIES}" \
  --deadly-corridor-latencies "${DEADLY_CORRIDOR_LATENCIES}" \
  --generated-root "${GENERATED_ROOT}" \
  --train-dir "${TRAIN_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "${DEVICE}" \
  "$@"
