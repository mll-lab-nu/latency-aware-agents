#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/workspace/latency-sensitive-bench

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
hf download latency-sensitive-bench/flappy_200ep \
  --repo-type dataset \
  --include "flappy_fix_latency_2_200ep/**" \
  --local-dir "${REPO_ROOT}/starVLA/data/raw/flappy_fix_latency_2_200ep"

# --- process data ---
# convert 脚本用相对路径，需在 starVLA/ 下运行；真实路径在 bash_scripts/gr00t/data_conversion/。
cd "${REPO_ROOT}/starVLA"
python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_flappy_to_starvla_lerobot.py \
  --dataset-name data/raw/flappy_fix_latency_2_200ep \
  --output-dir data/flappy_fix_latency_2_200ep/flappy_train__bridge \
  --action-carrier bridge \
  --force

# --- train (fix latency, not freeze ViT/visual connector + freeze LLM layers 0..27) ---
bash "${REPO_ROOT}/starVLA/commands/memory/train_flappy_openvla_last_8_layers_corrected.sh"

# --- upload checkpoints ---
# checkpoint 落在 starVLA/results/Checkpoints/<run_id>，run_id=flappy_fix_latency_2_200ep
hf upload latency-sensitive-bench/memory \
  "${REPO_ROOT}/starVLA/results/Checkpoints/flappy_fix_latency_2_200ep_last_8_layers_corrected" \
  ./flappy_fix_latency_2_200ep_last_8_layers_corrected \
  --exclude "wandb/**" \
  --repo-type model
