#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

required_variables=(
  CONFIG_PATH
  BASE_MODEL_DIR
  PRE_SFT_CHECKPOINT
  POST_SFT_CHECKPOINT
  TRAIN_DATASET_DIR
  VALIDATION_DATASET_DIR
  OUTPUT_DIR
  WANDB_ENTITY
  WANDB_PROJECT
  WANDB_RUN_NAME
)

for variable_name in "${required_variables[@]}"; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Missing required environment variable: ${variable_name}" >&2
    exit 1
  fi
done

python examples/rl_games/scripts/probe_wan_oft_temporal_latents.py \
  --config "${CONFIG_PATH}" \
  --base-model-dir "${BASE_MODEL_DIR}" \
  --pre-sft-checkpoint "${PRE_SFT_CHECKPOINT}" \
  --post-sft-checkpoint "${POST_SFT_CHECKPOINT}" \
  --train-dataset-dir "${TRAIN_DATASET_DIR}" \
  --validation-dataset-dir "${VALIDATION_DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-device "${MODEL_DEVICE:-cuda:0}" \
  --probe-device "${PROBE_DEVICE:-cuda:0}" \
  --extraction-batch-size "${EXTRACTION_BATCH_SIZE:-4}" \
  --max-train-episodes "${MAX_TRAIN_EPISODES:-20}" \
  --max-validation-episodes "${MAX_VALIDATION_EPISODES:-5}" \
  --image-sequence-length 5 \
  --maximum-exact-distance 4 \
  --flap-action-id 1 \
  --selection-seed "${SELECTION_SEED:-3047}" \
  --control-seed "${CONTROL_SEED:-6201}" \
  --vae-seed "${VAE_SEED:-9109}" \
  --probe-seed "${PROBE_SEED:-12109}" \
  --probe-epochs "${PROBE_EPOCHS:-50}" \
  --probe-batch-size "${PROBE_BATCH_SIZE:-512}" \
  --probe-learning-rate "${PROBE_LEARNING_RATE:-1e-3}" \
  --probe-weight-decay "${PROBE_WEIGHT_DECAY:-1e-4}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}"
