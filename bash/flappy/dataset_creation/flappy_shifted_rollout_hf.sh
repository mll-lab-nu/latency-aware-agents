#!/bin/bash
#SBATCH --job-name=flappy_mixed_teacher_rollout_hf
#SBATCH --output=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/flappy_mixed_teacher_rollout_hf_%j.log
#SBATCH --error=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/flappy_mixed_teacher_rollout_hf_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=tchafekar@umass.edu

set -euo pipefail

EXP_NAME="flappy_mixed_teacher_rollout_hf"
WORKSPACE="/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency"
EXP_DIR="${WORKSPACE}/distill/flappy"

PIPELINE_SCRIPT="${WORKSPACE}/scripts/rollouts_to_hf_dataset/build_flappy_mixed_rollout.py"
CONDA_ENV="/scratch/workspace/tchafekar_umass_edu-tchafekar/.conda/envs/flappy_bird_env2"

TEACHER_ROLLOUT_PATH="${EXP_DIR}/teacher_rollouts/flappy_queue_teacher_rollouts.pkl"
MIXED_ROLLOUT_ROOT="${EXP_DIR}/mixed_teacher_rollouts"
HF_OUTPUT_DIR="${EXP_DIR}/hf_flappy_mixed_teacher_rollout"
LOCAL_PARQUET_DIR="${HF_OUTPUT_DIR}/parquet"
REPO_ID="talha1503/flappy_mixed_teacher_rollout_parquet"

ENV_NAME="Flappy Bird"
ACTION_NAMES="NOOP,FLAP"
BASE_PROMPT_TEMPLATE="You are playing {env_name}. Pass through the pipe gaps and stay alive. Choose the action: {action_names}."

VAL_RATIO=0.2
SEED=42
BASE_SEED=0
FRAME_POLICY="latest"
REWARD_MODE="truncate"
LOCAL_TRAIN_NUM_SHARDS=16
LOCAL_VALIDATION_NUM_SHARDS=4
MAX_SHARD_SIZE="500MB"

LATENCIES=(0 1 2 3 4 5 10 15)
EPISODES_PER_LATENCY=0

rm -rf "${HF_OUTPUT_DIR}/images" "${LOCAL_PARQUET_DIR}"
mkdir -p "${EXP_DIR}" "${MIXED_ROLLOUT_ROOT}" "${HF_OUTPUT_DIR}" "${LOCAL_PARQUET_DIR}"
cd "${EXP_DIR}"

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "=== DEBUG ENV ==="
echo "EXP_NAME=${EXP_NAME}"
echo "EXP_DIR=${EXP_DIR}"
echo "PIPELINE_SCRIPT=${PIPELINE_SCRIPT}"
echo "TEACHER_ROLLOUT_PATH=${TEACHER_ROLLOUT_PATH}"
echo "MIXED_ROLLOUT_ROOT=${MIXED_ROLLOUT_ROOT}"
echo "HF_OUTPUT_DIR=${HF_OUTPUT_DIR}"
echo "REPO_ID=${REPO_ID}"
echo "LATENCIES=${LATENCIES[*]}"
echo "EPISODES_PER_LATENCY=${EPISODES_PER_LATENCY}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
which conda
which python
python -V
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import datasets, PIL, numpy; print('datasets:', datasets.__version__)"
python -c "from huggingface_hub import whoami; print('hf auth ok:', whoami()['name'])" || echo "HF auth check skipped/failed"
echo "=== END DEBUG ==="

if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
  echo "ERROR: pipeline script not found at ${PIPELINE_SCRIPT}"
  exit 1
fi

if [[ ! -f "${TEACHER_ROLLOUT_PATH}" ]]; then
  echo "ERROR: teacher rollout pickle not found at ${TEACHER_ROLLOUT_PATH}"
  exit 1
fi

PROMPT_TEMPLATE="${BASE_PROMPT_TEMPLATE} The action you choose now will be applied after {latency} steps."

python "${PIPELINE_SCRIPT}" \
  --teacher-rollout-path "${TEACHER_ROLLOUT_PATH}" \
  --mixed-output-root "${MIXED_ROLLOUT_ROOT}" \
  --latencies "${LATENCIES[@]}" \
  --episodes-per-latency ${EPISODES_PER_LATENCY} \
  --reward-mode "${REWARD_MODE}" \
  --seed ${SEED} \
  --base-seed ${BASE_SEED} \
  --output-dir "${HF_OUTPUT_DIR}" \
  --repo-id "${REPO_ID}" \
  --env-name "${ENV_NAME}" \
  --action-names "${ACTION_NAMES}" \
  --prompt-template "${PROMPT_TEMPLATE}" \
  --val-ratio ${VAL_RATIO} \
  --frame-policy "${FRAME_POLICY}" \
  --write-local-parquet \
  --local-parquet-dir "${LOCAL_PARQUET_DIR}" \
  --local-train-num-shards ${LOCAL_TRAIN_NUM_SHARDS} \
  --local-validation-num-shards ${LOCAL_VALIDATION_NUM_SHARDS} \
  --push-to-hub \
  --max-shard-size "${MAX_SHARD_SIZE}"

echo "Done. HF dataset images: ${HF_OUTPUT_DIR}/images"
echo "Done. Local parquet shards: ${LOCAL_PARQUET_DIR}"
echo "Done. Hub repo: ${REPO_ID}"
echo "Done. Mixed teacher rollout pickle: ${MIXED_ROLLOUT_ROOT}/mixed_teacher_rollouts.pkl"
