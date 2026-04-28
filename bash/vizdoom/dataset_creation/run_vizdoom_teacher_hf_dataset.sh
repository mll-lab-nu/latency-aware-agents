#!/bin/bash
#SBATCH --job-name=viz_basic_teacher_hf_dataset
#SBATCH --output=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/viz_basic_teacher_hf_dataset_%j.log
#SBATCH --error=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/viz_basic_teacher_hf_dataset_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=tchafekar@umass.edu

set -euo pipefail

EXP_NAME="viz_basic_teacher_hf_dataset"
WORKSPACE="/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency"
EXP_DIR="${WORKSPACE}/distill/vizdoom/viz_basic_queue_distill_image_teacher"
CONVERTER_SCRIPT="${WORKSPACE}/scripts/rollouts_to_hf_dataset/vizdoom_teacher_rollout_hf.py"
CONDA_ENV="/scratch/workspace/tchafekar_umass_edu-tchafekar/.conda/envs/flappy_bird_env2"

TEACHER_ROLLOUT_PATH="${EXP_DIR}/teacher_rollouts/vizdoom_basic_teacher_latency0_rollouts.pkl"
HF_OUTPUT_DIR="${EXP_DIR}/hf_vizdoom_basic_teacher"
LOCAL_PARQUET_DIR="${HF_OUTPUT_DIR}/parquet"
REPO_ID="talha1503/vizdoom_basic_teacher_parquet"

ENV_NAME="VizDoom Basic"
ACTION_NAMES="NOOP,MOVE_LEFT,MOVE_RIGHT,ATTACK"
PROMPT_TEMPLATE="You are playing {env_name}. Align with the monster, shoot it, and survive. Choose the next action: {action_names}."

VAL_RATIO=0.2
SEED=42
DEFAULT_FRAME_SKIP=0
DEFAULT_SEED_BASE=0
MAX_SHARD_SIZE="500MB"
LOCAL_TRAIN_NUM_SHARDS=32
LOCAL_VALIDATION_NUM_SHARDS=8

mkdir -p "${EXP_DIR}" "${HF_OUTPUT_DIR}" "${LOCAL_PARQUET_DIR}"
cd "${EXP_DIR}"

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== DEBUG ENV ==="
echo "EXP_NAME=${EXP_NAME}"
echo "EXP_DIR=${EXP_DIR}"
echo "CONVERTER_SCRIPT=${CONVERTER_SCRIPT}"
echo "TEACHER_ROLLOUT_PATH=${TEACHER_ROLLOUT_PATH}"
echo "HF_OUTPUT_DIR=${HF_OUTPUT_DIR}"
echo "REPO_ID=${REPO_ID}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
which conda
which python
python -V
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import datasets, PIL, numpy; print('datasets:', datasets.__version__)"
python -c "from huggingface_hub import whoami; print('hf auth ok:', whoami()['name'])" || echo "HF auth check skipped/failed; will rely on existing cached login or token"
echo "=== END DEBUG ==="

if [[ ! -f "${CONVERTER_SCRIPT}" ]]; then
  echo "ERROR: converter script not found at ${CONVERTER_SCRIPT}"
  exit 1
fi

if [[ ! -f "${TEACHER_ROLLOUT_PATH}" ]]; then
  echo "ERROR: teacher rollout pickle not found at ${TEACHER_ROLLOUT_PATH}"
  exit 1
fi

python "${CONVERTER_SCRIPT}" \
  --teacher-rollout-path "${TEACHER_ROLLOUT_PATH}" \
  --output-dir "${HF_OUTPUT_DIR}" \
  --repo-id "${REPO_ID}" \
  --env-name "${ENV_NAME}" \
  --action-names "${ACTION_NAMES}" \
  --prompt-template "${PROMPT_TEMPLATE}" \
  --val-ratio ${VAL_RATIO} \
  --seed ${SEED} \
  --default-frame-skip ${DEFAULT_FRAME_SKIP} \
  --default-seed-base ${DEFAULT_SEED_BASE} \
  --write-local-parquet \
  --local-parquet-dir "${LOCAL_PARQUET_DIR}" \
  --local-train-num-shards ${LOCAL_TRAIN_NUM_SHARDS} \
  --local-validation-num-shards ${LOCAL_VALIDATION_NUM_SHARDS} \
  --push-to-hub \
  --max-shard-size "${MAX_SHARD_SIZE}"


echo "Done. Dataset images: ${HF_OUTPUT_DIR}/images"
echo "Done. Local parquet shards: ${LOCAL_PARQUET_DIR}"
echo "Done. Hub repo: ${REPO_ID}"