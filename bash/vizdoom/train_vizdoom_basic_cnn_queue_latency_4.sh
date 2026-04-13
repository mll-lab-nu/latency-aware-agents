#!/bin/bash

#SBATCH --job-name=viz_basic_cnn_queue_latency_4
#SBATCH --output=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/viz_basic_cnn_queue_latency_4_%j.log
#SBATCH --error=/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/unity_logs/viz_basic_cnn_queue_latency_4_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=tchafekar@umass.edu

EXP_NAME="viz_basic_cnn_queue_latency_4"
WORKSPACE="/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency"
EXP_DIR="${WORKSPACE}/vizdoom_basic_models/${EXP_NAME}"
PY_SCRIPT="${WORKSPACE}/scripts/vizdoom_basic_queue_latency_ppo.py"
CONDA_ENV="/scratch/workspace/tchafekar_umass_edu-tchafekar/.conda/envs/flappy_bird_env2"
WORKER_TMP_ROOT="${EXP_DIR}/vizdoom_workers"

set -eo pipefail

mkdir -p "$EXP_DIR" "$WORKER_TMP_ROOT"
cd "$EXP_DIR"

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "=== DEBUG ENV ==="
echo "EXP_NAME=$EXP_NAME"
echo "EXP_DIR=$EXP_DIR"
which conda
which python
python -V
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import gymnasium; print('gymnasium ok:', gymnasium.__version__)"
python -c "import stable_baselines3; print('sb3 ok:', stable_baselines3.__version__)"
python -c "import wandb; print('wandb ok:', wandb.__version__)"
python -c "import vizdoom; print('vizdoom ok:', vizdoom.__version__)"
python - <<'PY'
import gymnasium as gym
from vizdoom import gymnasium_wrapper  # registers envs
env = gym.make('VizdoomBasic-v1', frame_skip=1)
print('made env:', env)
env.close()
PY
echo "=== END DEBUG ==="

# python "$PY_SCRIPT" \
#   --env-id VizdoomBasic-v1 \
#   --train-steps 100000000 \
#   --n-envs 16 \
#   --n-steps 2048 \
#   --batch-size 256 \
#   --ent-coef 0.01 \
#   --vizdoom-frame-skip 1 \
#   --max-buttons-pressed 1 \
#   --screen-size 84,84 \
#   --worker-tmp-root "$WORKER_TMP_ROOT" \
#   --grayscale \
#   --stack-frames 4 \
#   --checkpoint-freq 25000 \
#   --frame-skip-values 4 \
#   --latency-mode episode_fixed \
#   --latency-sampling-mode uniform_all \
#   --obs-mode image_only \
#   --save "${EXP_DIR}/${EXP_NAME}.zip" \
#   --outdir "${EXP_DIR}/exports" \
#   --wandb \
#   --wandb-project vizdoom_basic_rl_queue_latency \
#   --wandb-run-name "$EXP_NAME"

# Eval-only example:
python "$PY_SCRIPT" \
  --train-steps 0 \
  --load "/scratch/workspace/tchafekar_umass_edu-tchafekar/vla_latency/vizdoom_basic_models/viz_basic_cnn_queue_latency_4/viz_basic_cnn_queue_latency_4_4800000_steps.zip" \
  --obs-mode image_only \
  --eval-episodes 5 \
  --eval-frame-skip-values 0,1,2,3,4,5,10,15 \
  --eval-max-steps 5000 \
  --vizdoom-frame-skip 1 \
  --max-buttons-pressed 1 \
  --screen-size 84,84 \
  --worker-tmp-root "$WORKER_TMP_ROOT" \
  --grayscale \
  --stack-frames 4
