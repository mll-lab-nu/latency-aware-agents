#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

python -m latency_bench.cross_device.protocol aggregate \
  --spec configs/experiments/cross_device_10real_20sim.yaml
