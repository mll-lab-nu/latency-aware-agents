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

export CROSS_DEVICE_RUNTIME_ENV=openvla
export HF_PROFILE_REPO=latency-sensitive-bench/profiles
export HF_PROFILE_REPO_TYPE=dataset

POLICY=${POLICY:-openvla-flappy}
OUT=${OUT:-results/cross_device}
PROFILES_DIR=${PROFILES_DIR:-results/cross_device/profiles}

python -m scripts.cross_device.aggregate_profiles_from_instances \
  --root "$OUT" \
  --profiles-dir "$PROFILES_DIR" \
  --only-policy "$POLICY" \
  --upload
