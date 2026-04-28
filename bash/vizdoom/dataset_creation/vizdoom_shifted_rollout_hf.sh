#!/bin/bash
#SBATCH --job-name=viz_basic_shifted_rollout_hf
#SBATCH --output=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/viz_basic_shifted_rollout_hf_%j.log
#SBATCH --error=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/viz_basic_shifted_rollout_hf_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=tchafekar@umass.edu

set -euo pipefail

EXP_NAME="viz_basic_shifted_rollout_hf"
WORKSPACE="/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency"
EXP_DIR="${WORKSPACE}/distill/vizdoom/viz_basic_queue_distill_image_teacher"

SHIFT_SCRIPT="${WORKSPACE}/scripts/rollouts_to_hf_dataset/build_vizdoom_shifted_rollouts.py"
CONVERTER_SCRIPT="${WORKSPACE}/scripts/rollouts_to_hf_dataset/vizdoom_teacher_rollout_hf.py"
CONDA_ENV="/scratch/workspace/tchafekar_umass_edu-tchafekar/.conda/envs/flappy_bird_env2"

TEACHER_ROLLOUT_PATH="${EXP_DIR}/teacher_rollouts/vizdoom_basic_teacher_latency0_rollouts.pkl"
SHIFTED_ROLLOUT_ROOT="${EXP_DIR}/shifted_rollouts"
HF_ROOT_DIR="${EXP_DIR}/hf_shifted_rollouts"
REPO_PREFIX="talha1503/vizdoom_shifted_rollout"

ENV_NAME="VizDoom Basic"
ACTION_NAMES="NOOP,MOVE_LEFT,MOVE_RIGHT,ATTACK"
BASE_PROMPT_TEMPLATE="You are playing {env_name}. Align with the monster, shoot it, and survive. Choose the next action: {action_names}."

VAL_RATIO=0.2
SEED=42
DEFAULT_SEED_BASE=0
MAX_SHARD_SIZE="500MB"
LOCAL_TRAIN_NUM_SHARDS=32
LOCAL_VALIDATION_NUM_SHARDS=8
REWARD_MODE="truncate"

LATENCIES=(1 2 3 4 5 10 15)

mkdir -p "${EXP_DIR}" "${SHIFTED_ROLLOUT_ROOT}" "${HF_ROOT_DIR}"
cd "${EXP_DIR}"

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== DEBUG ENV ==="
echo "EXP_NAME=${EXP_NAME}"
echo "EXP_DIR=${EXP_DIR}"
echo "SHIFT_SCRIPT=${SHIFT_SCRIPT}"
echo "CONVERTER_SCRIPT=${CONVERTER_SCRIPT}"
echo "TEACHER_ROLLOUT_PATH=${TEACHER_ROLLOUT_PATH}"
echo "SHIFTED_ROLLOUT_ROOT=${SHIFTED_ROLLOUT_ROOT}"
echo "HF_ROOT_DIR=${HF_ROOT_DIR}"
echo "REPO_PREFIX=${REPO_PREFIX}"
echo "REWARD_MODE=${REWARD_MODE}"
echo "LATENCIES=${LATENCIES[*]}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
which conda
which python
python -V
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import datasets, PIL, numpy; print('datasets:', datasets.__version__)"
python -c "from huggingface_hub import whoami; print('hf auth ok:', whoami()['name'])" || echo "HF auth check skipped/failed; will rely on existing cached login or token"
echo "=== END DEBUG ==="

if [[ ! -f "${SHIFT_SCRIPT}" ]]; then
  echo "ERROR: shift script not found at ${SHIFT_SCRIPT}"
  exit 1
fi

if [[ ! -f "${CONVERTER_SCRIPT}" ]]; then
  echo "ERROR: converter script not found at ${CONVERTER_SCRIPT}"
  exit 1
fi

if [[ ! -f "${TEACHER_ROLLOUT_PATH}" ]]; then
  echo "ERROR: teacher rollout pickle not found at ${TEACHER_ROLLOUT_PATH}"
  exit 1
fi

python "${SHIFT_SCRIPT}" \
  --teacher-rollout-path "${TEACHER_ROLLOUT_PATH}" \
  --output-root "${SHIFTED_ROLLOUT_ROOT}" \
  --latencies "${LATENCIES[@]}" \
  --reward-mode "${REWARD_MODE}"

for LAT in "${LATENCIES[@]}"; do
  SHIFTED_ROLLOUT_PATH="${SHIFTED_ROLLOUT_ROOT}/latency_${LAT}/shifted_rollout_latency_${LAT}.pkl"
  HF_OUTPUT_DIR="${HF_ROOT_DIR}/latency_${LAT}"
  LOCAL_PARQUET_DIR="${HF_OUTPUT_DIR}/parquet"
  REPO_ID="${REPO_PREFIX}_latency_${LAT}"
  PROMPT_TEMPLATE="${BASE_PROMPT_TEMPLATE} There is an action delay of ${LAT} steps, so the action you choose now will be executed ${LAT} steps later."

  echo
  echo "============================================================"
  echo "Processing latency ${LAT}"
  echo "SHIFTED_ROLLOUT_PATH=${SHIFTED_ROLLOUT_PATH}"
  echo "HF_OUTPUT_DIR=${HF_OUTPUT_DIR}"
  echo "LOCAL_PARQUET_DIR=${LOCAL_PARQUET_DIR}"
  echo "REPO_ID=${REPO_ID}"
  echo "PROMPT_TEMPLATE=${PROMPT_TEMPLATE}"
  echo "============================================================"

  if [[ ! -f "${SHIFTED_ROLLOUT_PATH}" ]]; then
    echo "ERROR: shifted rollout pickle not found at ${SHIFTED_ROLLOUT_PATH}"
    exit 1
  fi

  rm -rf "${HF_OUTPUT_DIR}/images" "${LOCAL_PARQUET_DIR}"
  mkdir -p "${HF_OUTPUT_DIR}" "${LOCAL_PARQUET_DIR}"

  python "${CONVERTER_SCRIPT}" \
    --teacher-rollout-path "${SHIFTED_ROLLOUT_PATH}" \
    --output-dir "${HF_OUTPUT_DIR}" \
    --repo-id "${REPO_ID}" \
    --env-name "${ENV_NAME}" \
    --action-names "${ACTION_NAMES}" \
    --prompt-template "${PROMPT_TEMPLATE}" \
    --val-ratio ${VAL_RATIO} \
    --seed ${SEED} \
    --default-frame-skip ${LAT} \
    --default-seed-base ${DEFAULT_SEED_BASE} \
    --write-local-parquet \
    --local-parquet-dir "${LOCAL_PARQUET_DIR}" \
    --local-train-num-shards ${LOCAL_TRAIN_NUM_SHARDS} \
    --local-validation-num-shards ${LOCAL_VALIDATION_NUM_SHARDS} \
    --push-to-hub \
    --max-shard-size "${MAX_SHARD_SIZE}"

  echo "Done latency ${LAT}. Dataset images: ${HF_OUTPUT_DIR}/images"
  echo "Done latency ${LAT}. Local parquet shards: ${LOCAL_PARQUET_DIR}"
  echo "Done latency ${LAT}. Hub repo: ${REPO_ID}"
done

echo "All requested latencies completed."