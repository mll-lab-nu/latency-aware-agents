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

SPEC=configs/experiments/cross_device_protocol.yaml
POLICY=openvla-flappy
PROFILES_DIR=results/cross_device/profiles
AGGREGATE_PROFILES="${AGGREGATE_PROFILES:-1}"
export POLICY
export PROFILES_DIR

if [[ "$AGGREGATE_PROFILES" == "1" ]]; then
  ./aggregate_profiles.sh
fi

python -m scripts.cross_device.protocol sim-same \
  --spec "$SPEC" \
  --hardware auto \
  --only-policy "$POLICY" \
  --strict-runtime-env \
  --profiles-dir "$PROFILES_DIR" \
  --resume

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "$GPU_NAME" in
  *H200*|*B200*)
    python -m scripts.cross_device.protocol sim-cross \
      --spec "$SPEC" \
      --replay-hardware auto \
      --only-policy "$POLICY" \
      --strict-runtime-env \
      --profiles-dir "$PROFILES_DIR" \
      --resume
    ;;
  *)
    echo "skip sim-cross: replay host is not H200/B200 ($GPU_NAME)"
    ;;
esac
