#!/usr/bin/env bash
# Set VLA_ACCELERATE to the native starVLA environment's launcher.
# Usage: DATA_ROOT INIT_CHECKPOINT [native starVLA overrides...]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="$(realpath "$1")"
INIT_CHECKPOINT="$(realpath "$2")"
shift 2
cd "${REPO_ROOT}/third_party/starVLA"
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
exec "${VLA_ACCELERATE}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/MIKASA/train_files/starvla_qwenoft_intercept_grab_fast_h1.yaml \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix local/intercept_grab_fast_vla_v0_h1_train \
  --datasets.vla_data.eval_data_mix local/intercept_grab_fast_vla_v0_h1_val \
  --datasets.vla_data.custom_mixtures_path "${DATA_ROOT}/mikasa_intercept_grab_fast_h1.json" \
  --trainer.pretrained_checkpoint "${INIT_CHECKPOINT}" \
  --checkpoint.save_training_state true --checkpoint.save_pt_file true "$@"
