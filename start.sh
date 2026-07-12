#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export SPEC="${SPEC:-configs/experiments/cross_device_1real_3sim.yaml}"
export METHOD="${METHOD:-temporal}"

bash measure.sh
bash publish.sh measure
bash sim.sh
bash publish.sh sim
