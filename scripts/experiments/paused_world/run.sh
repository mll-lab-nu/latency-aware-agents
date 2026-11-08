#!/usr/bin/env bash
set -euo pipefail

# Evaluate the six profile-trained hist8 checkpoints with zero inference delay.
# Append "Current action latency is 0 raw frames (0.00 ms)." to each game prompt.
# Run on Linux/CUDA from any directory. Defaults reproduce the published
# episode cohort and batching: 100 episodes, 32 environments, two GPUs.
#
#   bash scripts/experiments/paused_world/run.sh
#   WORKSPACE_DIR=/mnt/workspace bash scripts/experiments/paused_world/run.sh
#   GPUS=2,3 OUTPUT_DIR=/mnt/results/paused_world bash scripts/experiments/paused_world/run.sh
#   SKIP_INSTALL=1 EPISODES=2 bash scripts/experiments/paused_world/run.sh
#
# The two-episode invocation is an execution check, not a formal result.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "${REPO_ROOT}")}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/paused_world/$(date -u +%Y%m%dT%H%M%SZ)}"
cd "${REPO_ROOT}"

case "${1:-}" in
    -h|--help)
        cat <<'EOF'
Usage: bash scripts/experiments/paused_world/run.sh

Install -> activate -> download -> six paused evaluations -> results table.
Targets Linux/CUDA; Flappy cap 3600 raw frames, Demon Attack cap 7200.
Every game prompt includes: Current action latency is 0 raw frames (0.00 ms).

Environment variables:
  WORKSPACE_DIR         Workspace directory (default: parent of this checkout)
  OUTPUT_DIR            New results directory (default: $WORKSPACE_DIR/outputs/paused_world/UTC_TIMESTAMP)
  GPUS                  CUDA device IDs (default: 0,1)
  EPISODES              Episodes per checkpoint (default: 100)
  PARALLEL_ENVS         Parallel game instances (default: 32)
  INFERENCE_BATCH_SIZE  Batch size per inference worker (default: 16)
  SKIP_INSTALL=1        Reuse the installed Conda environments

Each run requires a new output directory. Downloads reuse the Hugging Face cache.
EOF
        exit 0
        ;;
    "") ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
esac

GPUS="${GPUS:-0,1}"
EPISODES="${EPISODES:-100}"
PARALLEL_ENVS="${PARALLEL_ENVS:-32}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-16}"
MODEL_ENV="${STARVLA_CONDA_ENV_NAME:-starvla_rl_games_openvla}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

# 1. Reuse the maintained installer for both games and the shared Qwen stack.
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
    bash install.sh --env flappy,demon_attack --model openvla --tier use
fi

# install.sh can bootstrap Miniconda when the calling shell has no conda.
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
    CONDA_BASE="${HOME}/miniconda3"
fi

# 2. Activate the same model environment for all three architectures.
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${MODEL_ENV}"
set -u
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/starVLA${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

# CUDA renumbers the selected physical GPUs starting at zero.
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
INFERENCE_DEVICES=()
for index in "${!GPU_IDS[@]}"; do
    INFERENCE_DEVICES+=("cuda:${index}")
done

# 3. Download pinned checkpoints/backbone and prepare zero-delay configs/prompt maps.
python -m scripts.experiments.paused_world.experiment prepare \
    --output-dir "${OUTPUT_DIR}" \
    --episodes "${EPISODES}" \
    --parallel-envs "${PARALLEL_ENVS}" \
    --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
    --inference-devices "${INFERENCE_DEVICES[@]}"

nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > "${OUTPUT_DIR}/hardware.csv"
python -m pip freeze > "${OUTPUT_DIR}/packages.txt"

# 4. Evaluate one checkpoint at a time; preserve the evaluator's exit status.
while IFS= read -r config; do
    echo "[paused-world] Evaluating ${config}"
    python -m latency_bench eval --config "${config}" 2>&1 | tee "$(dirname "${config}")/eval.log"
done < "${OUTPUT_DIR}/eval_configs.txt"

# 5. Save CSV/JSON results and print the paused-versus-archived-profile table.
python -m scripts.experiments.paused_world.experiment summarize --output-dir "${OUTPUT_DIR}"
