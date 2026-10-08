#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/workspace/latency-sensitive-bench

# --- inject .env ---
set -a
source "${REPO_ROOT}/.env"
set +a
export HF_ENDPOINT=https://huggingface.co
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
DATASET_REVISION="15d17287246bd19940c837d6c8a5327564f969c5"
DATASET_PREFIX="latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps"
DATASET_CACHE="${REPO_ROOT}/third_party/starVLA/data/raw/.hf-benchmark-datasets/${DATASET_REVISION}"
RAW_DATA_ROOT="${DATASET_CACHE}/${DATASET_PREFIX}"
hf download MLL-Lab/LAGEN-datasets \
  --revision "${DATASET_REVISION}" \
  --repo-type dataset \
  --include "latency-aware/flappy/flappy_fix_latency_2_200ep_7k2steps/**" \
  --local-dir "${DATASET_CACHE}"

# --- process data ---
# convert 脚本用相对路径，需在 third_party/starVLA/ 下运行；真实路径在 bash_scripts/gr00t/data_conversion/。
cd "${REPO_ROOT}/third_party/starVLA"
python examples/rl_games/bash_scripts/gr00t/data_conversion/convert_flappy_to_starvla_lerobot.py \
  --dataset-name "${RAW_DATA_ROOT}" \
  --source-fps 30 \
  --obs-stride-raw-frames 1 \
  --source-latency-column latency_raw_frames \
  --target-latency-unit raw_frames \
  --output-dir data/flappy_fix_latency_2_200ep_7k2steps/flappy_train__bridge \
  --action-carrier bridge \
  --force

# cut down memory segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- train (fix latency, full tuning) ---
bash "${REPO_ROOT}/third_party/starVLA/commands/memory/train_flappy_openvla_kv_memory_flex.sh"

# --- upload checkpoints ---
# checkpoint 落在 third_party/starVLA/results/Checkpoints/<run_id>，run_id=flappy_fix_latency_2_200ep
RUN_DIR="${REPO_ROOT}/third_party/starVLA/results/Checkpoints/flappy_fix_latency_2_200ep_7k2steps_kv_memory_flex"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
compgen -G "${CHECKPOINT_DIR}/steps_*_model.safetensors" > /dev/null
find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name "steps_*_state" -exec rm -rf {} +

hf upload MLL-Lab/LAGEN-models \
  "${RUN_DIR}" \
  ./latency-aware/flappy/vla/starvla-qwenoft-h1/flappy_fix_latency_2_200ep_7k2steps_kv_memory_flex \
  --exclude "wandb/**" \
  --exclude "checkpoints/_initialization/**" \
  --exclude "checkpoints/*_state/**" \
  --exclude "**/optimizer.bin" \
  --exclude "**/*optim_states.pt" \
  --repo-type model
