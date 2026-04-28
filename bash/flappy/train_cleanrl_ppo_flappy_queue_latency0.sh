#!/bin/bash
#SBATCH --job-name=train_cleanrl_ppo_flappy_latency0
#SBATCH --output=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/train_cleanrl_ppo_flappy_latency0_%j.log
#SBATCH --error=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/train_cleanrl_ppo_flappy_latency0_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="vram16"
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=tchafekar@umass.edu

set -euo pipefail

PROJECT_DIR=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency
ENV_PATH=/scratch/workspace/tchafekar_umass_edu-tchafekar/.conda/envs/flappy_bird_env2
PY_SCRIPT=${PROJECT_DIR}/scripts/cleanrl_ppo_flappy_queue_latency.py

TRAIN_STEPS=50000000
N_ENVS=8
N_STEPS=2048
BATCH_SIZE=128
N_EPOCHS=10
TRAIN_FRAME_STRIDE=1

EXP_NAME=cleanrl_ppo_flappy_latency0_50m_rgbstack4_exp1
MODEL_DIR=${PROJECT_DIR}/models/${EXP_NAME}
MODEL_SAVE=${MODEL_DIR}/${EXP_NAME}.pt
ROLLOUT_DIR=${PROJECT_DIR}/flappy_data/${EXP_NAME}
LOG_DIR=${PROJECT_DIR}/unity_logs

mkdir -p "${MODEL_DIR}" "${ROLLOUT_DIR}" "${LOG_DIR}"

cd "${PROJECT_DIR}"

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

echo "=== DEBUG ENV ==="
date
hostname
which conda
which python
python -V
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"
python -c "import gymnasium; print('gymnasium:', gymnasium.__version__)"
python -c "import flappy_bird_gymnasium; print('flappy_bird_gymnasium ok')"
python -c "import wandb; print('wandb:', wandb.__version__)"
echo "EXP_NAME=${EXP_NAME}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
echo "N_ENVS=${N_ENVS}"
echo "N_STEPS=${N_STEPS}"
echo "EFFECTIVE_ROLLOUT_BATCH=$((N_ENVS * N_STEPS))"
echo "TRAIN_FRAME_STRIDE=${TRAIN_FRAME_STRIDE}"
echo "COLLECTION_STEPS_PER_UPDATE=$((N_STEPS * TRAIN_FRAME_STRIDE))"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MODEL_SAVE=${MODEL_SAVE}"
echo "ROLLOUT_DIR=${ROLLOUT_DIR}"
echo "=== END DEBUG ==="

# NOTE:
# - Default below uses RGB stacked frames, matching your SB3 Flappy image setup more closely: obs = 12x84x84.
# - Add --grayscale if you want Atari/CleanRL-style obs = 4x84x84, closer to DemonAttack.

python "${PY_SCRIPT}" \
  --train-steps "${TRAIN_STEPS}" \
  --n-envs "${N_ENVS}" \
  --n-steps "${N_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --n-epochs "${N_EPOCHS}" \
  --train-frame-stride "${TRAIN_FRAME_STRIDE}" \
  --learning-rate 2.5e-4 \
  --learning-rate-final 2.5e-6 \
  --clip-range 0.15 \
  --clip-range-final 0.025 \
  --ent-coef 0.01 \
  --vf-coef 0.5 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --max-grad-norm 0.5 \
  --fps 30 \
  --image-size 84 \
  --stack-frames 4 \
  --obs-mode image_only \
  --frame-skip-values 0 \
  --latency-mode episode_fixed \
  --latency-sampling-mode uniform_all \
  --mid-eval \
  --mid-eval-freq 250000 \
  --mid-eval-episodes 20 \
  --mid-eval-frame-skip-values 0 \
  --eval-episodes 20 \
  --eval-frame-skip-values 0,1,2,3,4,5,10,15 \
  --eval-max-steps 5000 \
  --save "${MODEL_SAVE}" \
  --outdir "${ROLLOUT_DIR}" \
  --checkpoint-freq 250000 \
  --wandb \
  --wandb-project flappy_cleanrl_queue_latency \
  --wandb-run-name "${EXP_NAME}"

# Optional eval-only example for a checkpoint produced by this CleanRL script:
# python "${PY_SCRIPT}" \
#   --train-steps 0 \
#   --load "${MODEL_DIR}/${EXP_NAME}_2250000_steps.pt" \
#   --save "${MODEL_SAVE}" \
#   --obs-mode image_only \
#   --eval-episodes 5 \
#   --eval-frame-skip-values 0,1,2,3,4,5,10,15 \
#   --eval-max-steps 5000 \
#   --fps 30 \
#   --image-size 84 \
#   --stack-frames 4 \
#   --frame-skip-values 0 \
#   --outdir "${ROLLOUT_DIR}/eval_only"
