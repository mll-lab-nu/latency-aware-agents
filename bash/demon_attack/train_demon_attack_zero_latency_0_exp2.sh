#!/bin/bash
#SBATCH --job-name=train_ppo_demon_attack_oldstyle_raw_eval_with_env_fs_4_terminal_on_life_loss
#SBATCH --output=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/train_ppo_demon_attack_oldstyle_raw_eval_terminal_on_life_loss_%j.log
#SBATCH --error=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/train_ppo_demon_attack_oldstyle_raw_eval_terminal_on_life_loss_%j.err
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
PY_SCRIPT=${PROJECT_DIR}/scripts/demon_attack_queue_latency.py

# Old-style PPO Atari baseline params
TRAIN_STEPS=10000000
N_ENVS=8
N_STEPS=128
BATCH_SIZE=256
N_EPOCHS=4

EXP_NAME=ppo_demon_attack_queue_latency_0_oldstyle_10m_trainclip_evalraw_envfs4_terminal_on_life_loss
MODEL_DIR=${PROJECT_DIR}/models/${EXP_NAME}
MODEL_SAVE=${MODEL_DIR}/${EXP_NAME}.zip
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
python -c "import gymnasium; print('gymnasium:', gymnasium.__version__)"
python -c "import stable_baselines3; print('sb3:', stable_baselines3.__version__)"
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
  --terminal-on-life-loss \
  --gae-lambda 0.95 \
  --max-grad-norm 0.5 \
  --screen-size 84 \
  --noop-max 30 \
  --stack-frames 4 \
  --clip-reward \
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
  --eval-max-steps 30000 \
  --save "${MODEL_SAVE}" \
  --outdir "${ROLLOUT_DIR}" \
  --checkpoint-freq 250000 \
  --wandb \
  --wandb-project demon_attack_rl_queue_latency \
  --wandb-run-name "${EXP_NAME}"