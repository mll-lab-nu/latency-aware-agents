#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/miniforge3)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ -d /venv/starvla_rl_games_openvla ]]; then
  conda activate /venv/starvla_rl_games_openvla
else
  conda activate starvla_rl_games_openvla
fi

python -m latency_bench.cross_device.protocol sim \
  --spec configs/experiments/cross_device_10real_20sim.yaml \
  --hardware auto \
  --all-target-profiles \
  --latency-method temporal \
  --resume
