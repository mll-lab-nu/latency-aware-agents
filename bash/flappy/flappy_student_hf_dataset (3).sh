#!/bin/bash
#SBATCH --job-name=flappy_student_hf_dataset
#SBATCH --output=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/flappy_student_hf_dataset_%j.log
#SBATCH --error=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/flappy_student_hf_dataset_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=tchafekar@umass.edu

set -euo pipefail

EXP_NAME="flappy_student_hf_dataset"
WORKSPACE="/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency"
EXP_DIR="${WORKSPACE}/distill/flappy"

CONVERTER_SCRIPT="${WORKSPACE}/scripts/rollouts_to_hf_dataset/flappy_shifted_student_hf.py"
CONDA_ENV="/scratch/workspace/tchafekar_umass_edu-tchafekar/.conda/envs/flappy_bird_env2"

SHIFTED_DATASET_ROOT="${EXP_DIR}/shifted_datasets/ppo_image_queue_latency_0_3750000_steps"
HF_ROOT_DIR="${EXP_DIR}/hf_shifted_students"
REPO_PREFIX="talha1503/flappy_student_shifted"

ENV_NAME="Flappy Bird"
ACTION_NAMES="NOOP,FLAP"
PROMPT_TEMPLATE="You are playing {env_name}. Pass through the pipe gaps and stay alive. Choose the action: {action_names}."

VAL_RATIO=0.2
SEED=42
MAX_SHARD_SIZE="500MB"
LOCAL_TRAIN_NUM_SHARDS=16
LOCAL_VALIDATION_NUM_SHARDS=4
FRAME_POLICY="latest"

# Change this list as needed.
LATENCIES=(1 2 3 4 5 10 15)

mkdir -p "${EXP_DIR}" "${HF_ROOT_DIR}"
cd "${EXP_DIR}"

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== DEBUG ENV ==="
echo "EXP_NAME=${EXP_NAME}"
echo "EXP_DIR=${EXP_DIR}"
echo "CONVERTER_SCRIPT=${CONVERTER_SCRIPT}"
echo "SHIFTED_DATASET_ROOT=${SHIFTED_DATASET_ROOT}"
echo "HF_ROOT_DIR=${HF_ROOT_DIR}"
echo "REPO_PREFIX=${REPO_PREFIX}"
echo "FRAME_POLICY=${FRAME_POLICY}"
echo "LATENCIES=${LATENCIES[*]}"
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

if [[ ! -d "${SHIFTED_DATASET_ROOT}" ]]; then
  echo "ERROR: shifted dataset root not found at ${SHIFTED_DATASET_ROOT}"
  exit 1
fi

for LAT in "${LATENCIES[@]}"; do
  SHIFTED_DATASET_DIR="${SHIFTED_DATASET_ROOT}/latency_${LAT}"
  HF_OUTPUT_DIR="${HF_ROOT_DIR}/latency_${LAT}"
  LOCAL_PARQUET_DIR="${HF_OUTPUT_DIR}/parquet"
  REPO_ID="${REPO_PREFIX}_latency_${LAT}"

  echo
  echo "============================================================"
  echo "Processing latency ${LAT}"
  echo "SHIFTED_DATASET_DIR=${SHIFTED_DATASET_DIR}"
  echo "HF_OUTPUT_DIR=${HF_OUTPUT_DIR}"
  echo "LOCAL_PARQUET_DIR=${LOCAL_PARQUET_DIR}"
  echo "REPO_ID=${REPO_ID}"
  echo "============================================================"

  if [[ ! -d "${SHIFTED_DATASET_DIR}" ]]; then
    echo "ERROR: shifted dataset dir not found at ${SHIFTED_DATASET_DIR}"
    exit 1
  fi

  rm -rf "${HF_OUTPUT_DIR}/images" "${HF_OUTPUT_DIR}/history_strips" "${LOCAL_PARQUET_DIR}"
  mkdir -p "${HF_OUTPUT_DIR}" "${LOCAL_PARQUET_DIR}"

  python "${CONVERTER_SCRIPT}" \
    --shifted-dataset-dir "${SHIFTED_DATASET_DIR}" \
    --output-dir "${HF_OUTPUT_DIR}" \
    --repo-id "${REPO_ID}" \
    --env-name "${ENV_NAME}" \
    --action-names "${ACTION_NAMES}" \
    --prompt-template "${PROMPT_TEMPLATE}" \
    --val-ratio ${VAL_RATIO} \
    --seed ${SEED} \
    --frame-policy "${FRAME_POLICY}" \
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
