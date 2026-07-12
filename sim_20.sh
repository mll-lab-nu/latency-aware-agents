#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
SPEC=configs/experiments/cross_device_10real_20sim.yaml METHOD=temporal bash sim.sh
