#!/usr/bin/env bash
set -euo pipefail

python scripts/generate_sample_factory_same_latency_sweep.py \
  --envs flappy,demon_attack,deadly_corridor \
  --seeds 0 \
  --flappy-latencies 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --demon-attack-latencies 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --deadly-corridor-latencies 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --generated-root configs/train \
  --train-dir results/checkpoints \
  --output-root outputs \
  --device gpu \
  --restart-behavior resume \
  --eval-episodes 100
