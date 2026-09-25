#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# --- inject .env ---
set -a
source "${REPO_ROOT}/.env"
set +a
echo "HF_TOKEN set: ${HF_TOKEN:+yes}    WANDB_API_KEY set: ${WANDB_API_KEY:+yes}"

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
hf download placeholder/flappy_200ep \
  --repo-type dataset \
  --include "flappy_fix_latency_2_200ep/**" \
  --local-dir "${REPO_ROOT}/third_party/starVLA/data/raw/flappy_fix_latency_2_200ep"

# --- process data ---
# The converter uses relative paths and must run from third_party/starVLA.
cd "${REPO_ROOT}/third_party/starVLA"
python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_flappy_to_starvla_lerobot.py \
  --dataset-name data/raw/flappy_fix_latency_2_200ep \
  --source-fps 30 \
  --obs-stride-raw-frames 1 \
  --source-latency-column latency_raw_frames \
  --target-latency-unit raw_frames \
  --output-dir data/flappy_fix_latency_2_200ep/flappy_train__bridge \
  --action-carrier bridge \
  --force

# --- train (top eight LLM layers and action head) ---
bash "${REPO_ROOT}/third_party/starVLA/commands/memory/train_flappy_openvla_last_8_layers.sh"

# --- upload checkpoints ---
# The published checkpoint keeps its historical run ID.
RUN_DIR="${REPO_ROOT}/third_party/starVLA/results/Checkpoints/flappy_fix_latency_2_200ep_last_8_layers_corrected_v2"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
compgen -G "${CHECKPOINT_DIR}/steps_*_model.safetensors" > /dev/null
find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name "steps_*_state" -exec rm -rf {} +

hf upload placeholder/memory \
  "${RUN_DIR}" \
  ./flappy_fix_latency_2_200ep_last_8_layers_corrected_v2 \
  --exclude "wandb/**" \
  --exclude "checkpoints/_initialization/**" \
  --exclude "checkpoints/*_state/**" \
  --exclude "**/optimizer.bin" \
  --exclude "**/*optim_states.pt" \
  --repo-type model
