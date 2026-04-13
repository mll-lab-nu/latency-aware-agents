#!/bin/bash

#SBATCH --job-name=ppo_flappy_mlp_queue_latency_eval_0_1_2_3_4_5_10_15
#SBATCH --output=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/ppo_flappy_mlp_queue_latency_%j.log
#SBATCH --error=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/ppo_flappy_mlp_queue_latency_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=tchafekar@umass.edu

set -eo pipefail

PROJECT_DIR=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency
ENV_PATH=/scratch/workspace/tchafekar_umass_edu-tchafekar/.conda/envs/flappy_bird_env2
PY_SCRIPT=${PROJECT_DIR}/flappy_queue_latency_cnn.py
MODEL_SAVE=${PROJECT_DIR}/models/ppo_image_queue_latency_0_1_2_3_4_5_basic.zip
ROLLOUT_DIR=${PROJECT_DIR}/flappy_data/ppo_image_queue_latency_basic

cd "${PROJECT_DIR}"
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

echo "=== DEBUG ENV ==="
which conda
which python
python -V
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import gymnasium; print('gymnasium ok:', gymnasium.__version__)"
python -c "import stable_baselines3; print('sb3 ok:', stable_baselines3.__version__)"
python -c "import wandb; print('wandb ok:', wandb.__version__)"
echo "=== END DEBUG ==="

python ./scripts/flappy_queue_latency_mlp.py \
  --train-steps 0 \
  --load /scratch/workspace/tchafekar_umass_edu-tchafekar/vla_generalization/models/ppo_numeric_descriptive_latency_training_0,1,2,3,4,5_EXP1.zip \
  --normalize-obs \
  --use-vecnormalize-reward \
  --outdir /scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/outputs \
  --eval-episodes 5 \
  --eval-frame-skip-values 0,1,2,3,4,5,10,15 \
  --obs-mode state_only \
  --include-latency-obs \
  --eval-max-steps 5000 \
  --fps 30


