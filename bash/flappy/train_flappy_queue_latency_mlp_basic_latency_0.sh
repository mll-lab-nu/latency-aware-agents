#!/bin/bash

#SBATCH --job-name=ppo_flappy_mlp_queue_latency_0
#SBATCH --output=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/ppo_flappy_mlp_queue_latency_%j.log
#SBATCH --error=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/ppo_flappy_mlp_queue_latency_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=tchafekar@umass.edu

set -eo pipefail

PROJECT_DIR=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency
ENV_PATH=/scratch/workspace/tchafekar_umass_edu-tchafekar/.conda/envs/flappy_bird_env2
PY_SCRIPT=${PROJECT_DIR}/scripts/flappy_queue_latency_mlp.py
MODEL_SAVE=${PROJECT_DIR}/models/ppo_numeric_queue_latency_0/ppo_numeric_queue_latency_0.zip
ROLLOUT_DIR=${PROJECT_DIR}/flappy_data/ppo_numeric_queue_latency_basic

cd "${PROJECT_DIR}"
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

echo "=== DEBUG ENV ==="
which conda
which python
python -V
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import gymnasium; print('gymnasium ok:', gymnasium.__version__)"
python -c "import stable_baselines3; print('sb3 ok:', stable_baselines3.__version__)"
python -c "import wandb; print('wandb ok:', wandb.__version__)"
echo "=== END DEBUG ==="

python "${PY_SCRIPT}" \
  --train-steps 25000000 \
  --n-envs 16 \
  --n-steps 2048 \
  --batch-size 256 \
  --n-epochs 10 \
  --ent-coef 0.01 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --fps 30 \
  --obs-mode state_only \
  --include-latency-obs \
  --use-vecnormalize-reward \
  --frame-skip-values 0 \
  --eval-frame-skip-values 0,1,2,3,4,5,10 \
  --eval-max-steps 5000 \
  --latency-mode episode_fixed \
  --latency-sampling-mode uniform_all \
  --activation tanh \
  --outdir "${ROLLOUT_DIR}" \
  --save "${MODEL_SAVE}" \
  --wandb \
  --wandb-project flappy_rl_numeric_queue_latency \
  --wandb-run-name ppo_mlp_queue_latency_0

# python "${PY_SCRIPT}" \
#   --train-steps 0 \
#   --load "${MODEL_SAVE}" \
#   --obs-mode state_only \
#   --include-latency-obs \
#   --use-vecnormalize-reward \
#   --eval-episodes 5 \
#   --eval-frame-skip-values 0,1,2,3,4,5,10,15 \
#   --eval-max-steps 5000 \
#   --fps 30