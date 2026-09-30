#!/usr/bin/env bash
set -euo pipefail

if (($#)); then
  SEEDS=("$@")
else
  SEEDS=(10 11 12 13 14)
fi
CONFIG_ROOT="configs/train/memory/small_model/inverted_pendulum/latency_024/frame_stack"

train_inverted_pendulum_latency_0() {
  for seed in "${SEEDS[@]}"; do
    python -m training.train \
      --config "${CONFIG_ROOT}/fixed_l0/fs1/seed_${seed}/train.yaml"
  done
}

train_inverted_pendulum_latency_2() {
  for seed in "${SEEDS[@]}"; do
    python -m training.train \
      --config "${CONFIG_ROOT}/fixed_l2/fs1/seed_${seed}/train.yaml"
  done
}

train_inverted_pendulum_latency_4() {
  for seed in "${SEEDS[@]}"; do
    python -m training.train \
      --config "${CONFIG_ROOT}/fixed_l4/fs1/seed_${seed}/train.yaml"
  done
}

case "${LATENCY:-all}" in
  0|l0|L0)
    train_inverted_pendulum_latency_0
    ;;
  2|l2|L2)
    train_inverted_pendulum_latency_2
    ;;
  4|l4|L4)
    train_inverted_pendulum_latency_4
    ;;
  all)
    train_inverted_pendulum_latency_0
    train_inverted_pendulum_latency_2
    train_inverted_pendulum_latency_4
    ;;
  *)
    echo "LATENCY must be one of: 0, 2, 4, all" >&2
    exit 2
    ;;
esac
