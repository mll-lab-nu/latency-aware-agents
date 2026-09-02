#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GR00T_ROOT="${REPO_ROOT}/third_party/Isaac-GR00T"
BASE_MODEL="${REPO_ROOT}/outputs/official_gr00t/models/GR00T-N1.7-3B"
DATASET="${REPO_ROOT}/outputs/official_gr00t/extreme_parkour_go2/lerobot"
OUTPUT="${REPO_ROOT}/outputs/official_gr00t/extreme_parkour_go2_latent_h1_5k"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=0,1 NO_ALBUMENTATIONS_UPDATE=1 \
  "${GR00T_ROOT}/.venv/bin/torchrun" \
  --standalone \
  --nproc_per_node=2 \
  "${GR00T_ROOT}/gr00t/experiment/launch_finetune.py" \
  --base-model-path "${BASE_MODEL}" \
  --dataset-path "${DATASET}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "${REPO_ROOT}/latency_bench/policy/official_gr00t_parkour_modality.py" \
  --num-gpus 2 \
  --output-dir "${OUTPUT}" \
  --max-state-dim 53 \
  --action-horizon 1 \
  --fixed-action-noise-seed 0 \
  --one-step-action-regression \
  --action-head-dropout 0 \
  --state-dropout-prob 0 \
  --no-use-percentiles \
  --tune-visual \
  --load-bf16 \
  --no-backbone-trainable-params-fp32 \
  --optim adamw_torch_fused \
  --learning-rate 0.0001 \
  --warmup-ratio 0.025 \
  --max-steps 5000 \
  --global-batch-size 128 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 8 \
  --save-steps 500 \
  --save-total-limit 2 \
  --use-wandb
