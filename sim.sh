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

export HF_PROFILE_REPO=latency-sensitive-bench/profiles
export HF_PROFILE_REPO_TYPE=dataset

SPEC=configs/experiments/cross_device_protocol.yaml
POLICY=openvla-flappy
PROFILES_DIR=results/cross_device/profiles
# Space-separated cross-device targets to replay on this host, each
# <device_id>/<instance_id>. Empty = same-device replay only.
SIM_TARGETS="${SIM_TARGETS:-}"
export POLICY
export PROFILES_DIR

# Same-device replay: this machine reads its own profile.
python -m latency_bench.cross_device.protocol sim \
  --spec "$SPEC" \
  --hardware auto \
  --only-policy "$POLICY" \
  --profiles-dir "$PROFILES_DIR" \
  --resume

# Cross-device replay: one run per explicitly requested target.
for TARGET in $SIM_TARGETS; do
  python -m latency_bench.cross_device.protocol sim \
    --spec "$SPEC" \
    --hardware auto \
    --target "$TARGET" \
    --only-policy "$POLICY" \
    --profiles-dir "$PROFILES_DIR" \
    --resume
done
