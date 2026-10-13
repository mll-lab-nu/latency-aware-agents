#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

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

# --- download data (latency_2) ---
hf download MLL-Lab/LAGEN-datasets \
  --repo-type dataset \
  --revision 15d17287246bd19940c837d6c8a5327564f969c5 \
  --include "latency-aware/flappy/flappy_200ep/flappy_fix_latency_2_200ep/**" \
  --local-dir "${REPO_ROOT}/third_party/starVLA/data/raw/latency-transfer"

# --- process data ---
# The converter uses relative paths and must run from third_party/starVLA.
cd "${REPO_ROOT}/third_party/starVLA"
python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_flappy_to_starvla_lerobot.py \
  --dataset-name data/raw/latency-transfer/latency-aware/flappy/flappy_200ep/flappy_fix_latency_2_200ep \
  --source-fps 30 \
  --obs-stride-raw-frames 1 \
  --source-latency-column latency_raw_frames \
  --target-latency-unit raw_frames \
  --output-dir data/flappy_fix_latency_2_200ep/flappy_train__bridge \
  --action-carrier bridge \
  --force

# --- train (vision stack, top eight LLM layers, and action head) ---
bash "${REPO_ROOT}/third_party/starVLA/commands/memory/train_flappy_openvla_last_8_layers_with_vision.sh"
