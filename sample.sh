#!/usr/bin/env bash
set -euo pipefail

# python scripts/generate_sample_factory_same_latency_sweep.py \
#   --envs flappy,demon_attack,deadly_corridor \
#   --seeds 0 \
#   --flappy-latencies 0 \
#   --demon-attack-latencies 0 \
#   --deadly-corridor-latencies 0 \
#   --generated-root configs/train/one_run \
#   --train-dir results/checkpoints/sample_factory_runs \
#   --output-root outputs/latency_sweeps \
#   --device gpu \
#   --restart-behavior resume \
#   --eval-episodes 10

# python -m training.train --config configs/train/one_run/flappy/latency_0/seed_0/train.yaml
# python -m latency_bench.run --config configs/train/one_run/flappy/latency_0/seed_0/latency_bench_eval.yaml

python -m training.train --config configs/train/one_run/demon_attack/latency_0/seed_0/train.yaml
# python -m latency_bench.run --config configs/train/one_run/demon_attack/latency_0/seed_0/latency_bench_eval.yaml

# python -m training.train --config configs/train/one_run/deadly_corridor/latency_0/seed_0/train.yaml
python -m latency_bench.run --config configs/train/one_run/deadly_corridor/latency_0/seed_0/latency_bench_eval.yaml
