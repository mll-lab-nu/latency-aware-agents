#!/bin/bash
#SBATCH --job-name=train_envpool_ppo_demon_attack_envfs1_lifeloss
#SBATCH --output=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/train_envpool_ppo_demon_attack_envfs1_lifeloss_%j.log
#SBATCH --error=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/train_envpool_ppo_demon_attack_envfs1_lifeloss_%j.err
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
PY_SCRIPT=${PROJECT_DIR}/scripts/cleanrl_ppo_demon_attack_envpool.py

# Longer fs=1 run: 40M fs=1 decision steps ~= 40M raw ALE frames.
# This is comparable raw-frame exposure to 10M steps with native frame skip 4.
TRAIN_STEPS=40000000
N_ENVS=32
N_STEPS=128
BATCH_SIZE=1024
N_EPOCHS=4

EXP_NAME=envpool_cleanrl_ppo_demon_attack_latency0_40m_envfs1_trainclip_lifeloss_evalraw_exp1
MODEL_DIR=${PROJECT_DIR}/models/${EXP_NAME}
MODEL_SAVE=${MODEL_DIR}/${EXP_NAME}.pt
ROLLOUT_DIR=${PROJECT_DIR}/demon_attack_data/${EXP_NAME}
LOG_DIR=${PROJECT_DIR}/unity_logs

mkdir -p "${MODEL_DIR}" "${ROLLOUT_DIR}" "${LOG_DIR}"

cd "${PROJECT_DIR}"

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

echo "=== DEBUG ENV ==="
date
hostname
which python
python -V
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'device_count:', torch.cuda.device_count())"
python -c "import envpool; print('envpool:', envpool.__version__)"
python -c "import ale_py; print('ale_py ok')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "EXP_NAME=${EXP_NAME}"
echo "MODEL_SAVE=${MODEL_SAVE}"
echo "PY_SCRIPT=${PY_SCRIPT}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
echo "N_ENVS=${N_ENVS}"
echo "N_STEPS=${N_STEPS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "=== END DEBUG ==="

python "${PY_SCRIPT}" \
  --env-id ALE/DemonAttack-v5 \
  --train-steps "${TRAIN_STEPS}" \
  --n-envs "${N_ENVS}" \
  --n-steps "${N_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --n-epochs "${N_EPOCHS}" \
  --learning-rate 2.5e-4 \
  --learning-rate-final 0.0 \
  --clip-range 0.1 \
  --clip-range-final 0.1 \
  --ent-coef 0.01 \
  --vf-coef 0.5 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --max-grad-norm 0.5 \
  --native-frame-skip 1 \
  --screen-size 84 \
  --noop-max 30 \
  --stack-frames 4 \
  --clip-reward \
  --terminal-on-life-loss \
  --eval-raw-reward \
  --obs-mode image_only \
  --frame-skip-values 0 \
  --latency-mode episode_fixed \
  --latency-sampling-mode uniform_all \
  --mid-eval \
  --mid-eval-freq 250000 \
  --mid-eval-episodes 20 \
  --mid-eval-frame-skip-values 0 \
  --eval-episodes 20 \
  --eval-frame-skip-values 0,1,2,3,4,5,10,15,40 \
  --eval-max-steps 120000 \
  --save "${MODEL_SAVE}" \
  --outdir "${ROLLOUT_DIR}" \
  --checkpoint-freq 250000 \
  --wandb \
  --wandb-project demon_attack_rl_queue_latency \
  --wandb-run-name "${EXP_NAME}"
