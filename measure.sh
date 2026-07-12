#!/usr/bin/env bash
set -euo pipefail

# run from repo root so relative configs/results paths resolve
cd "$(dirname "${BASH_SOURCE[0]}")"

# non-interactive bash doesn't auto-load conda's shell hook
CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/miniforge3)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# vast boxes use a prefix env under /venv/; plain conda hosts use a named env
if [[ -d /venv/starvla_rl_games_openvla ]]; then
  conda activate /venv/starvla_rl_games_openvla
else
  conda activate starvla_rl_games_openvla
fi

SPEC="${SPEC:-configs/experiments/cross_device_1real_3sim.yaml}"

python -m latency_bench.cross_device.protocol measure \
  --spec "$SPEC" \
  --hardware auto \
  --resume
