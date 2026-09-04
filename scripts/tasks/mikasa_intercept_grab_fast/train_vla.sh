#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <accepted-ppo-checkpoint>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MIKASA_ROOT="${REPO_ROOT}/third_party/MIKASA-Robo"
MIKASA_PY="${MIKASA_ROOT}/.venv/bin/python"
TASK_ROOT="${MIKASA_TASK_ROOT:-${REPO_ROOT}/outputs/tasks/mikasa_intercept_grab_fast}"
STRICT_ROOT="${TASK_ROOT}/strict_teacher"
DATASET_ROOT="${TASK_ROOT}/dataset"
VLA_ROOT="${REPO_ROOT}/third_party/starVLA"
VLA_PY="${VLA_PY:-python}"
VLA_PORT="${VLA_PORT:-10093}"
RUN_ID="${MIKASA_RUN_ID:-mikasa_grabfast_h1_gb64_mb8_ga8_expandable}"
TEACHER_CHECKPOINT="$1"

cd "${REPO_ROOT}"
bash scripts/mikasa/install.sh
export PYTHONPATH="${REPO_ROOT}:${MIKASA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PPO_CHECKPOINT="${STRICT_ROOT}/oracle_checkpoints/InterceptGrabFast-VLA-v0/final_success_ckpt.pt"
mkdir -p "$(dirname "${PPO_CHECKPOINT}")"
cp "${TEACHER_CHECKPOINT}" "${PPO_CHECKPOINT}"

"${MIKASA_PY}" scripts/mikasa/evaluate.py \
  --policy ppo \
  --checkpoint "${PPO_CHECKPOINT}" \
  --episodes 100 \
  --output-dir "${TASK_ROOT}/ppo_teacher_strict_eval"
"${MIKASA_PY}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); n=sum(d["successes"]); print(f"{n}/100"); assert n >= 95' \
  "${TASK_ROOT}/ppo_teacher_strict_eval/summary.json"

"${MIKASA_PY}" \
  "${MIKASA_ROOT}/mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py" \
  --env-id InterceptGrabFast-VLA-v0 \
  --path-to-save-data "${DATASET_ROOT}" \
  --ckpt-dir "${STRICT_ROOT}" \
  --num-train-data 275

uv run --project "${MIKASA_ROOT}/utils/convert_npz_to_rlds/rlds_dataset_builder" \
  python "${MIKASA_ROOT}/utils/convert_npz_to_rlds/convert_npz_task_to_rlds.py" \
  --task InterceptGrabFast-VLA-v0 \
  --repo-root "${MIKASA_ROOT}" \
  --npz-root "${DATASET_ROOT}/data_npz" \
  --rlds-root "${DATASET_ROOT}/data_rlds"

uv run --project "${MIKASA_ROOT}/utils/convert_rlds_to_lerobot" \
  python "${MIKASA_ROOT}/utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py" \
  --task InterceptGrabFast-VLA-v0 \
  --repo-root "${MIKASA_ROOT}" \
  --rlds-root "${DATASET_ROOT}/data_rlds" \
  --lerobot-root "${DATASET_ROOT}/data_lerobot" \
  --repo-id-template 'local/{task}_h1_train' \
  --fps 20 \
  --split 'train[:250]'

uv run --project "${MIKASA_ROOT}/utils/convert_rlds_to_lerobot" \
  python "${MIKASA_ROOT}/utils/convert_rlds_to_lerobot/convert_rlds_to_lerobot.py" \
  --task InterceptGrabFast-VLA-v0 \
  --repo-root "${MIKASA_ROOT}" \
  --rlds-root "${DATASET_ROOT}/data_rlds" \
  --lerobot-root "${DATASET_ROOT}/data_lerobot" \
  --repo-id-template 'local/{task}_h1_val' \
  --fps 20 \
  --split 'train[250:275]'

"${MIKASA_PY}" scripts/mikasa/prepare_starvla_dataset.py \
  --data-root "${DATASET_ROOT}/data_lerobot" \
  --train-name local/intercept_grab_fast_vla_v0_h1_train \
  --val-name local/intercept_grab_fast_vla_v0_h1_val

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
INIT_CKPT="${MIKASA_INIT_CKPT:-${VLA_ROOT}/playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1/checkpoints/steps_5000_pytorch_model.pt}"
DATA_ROOT="${DATASET_ROOT}/data_lerobot"
cd "${VLA_ROOT}"
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/MIKASA/train_files/starvla_qwenoft_intercept_grab_fast_h1.yaml \
  --run_id "${RUN_ID}" \
  --wandb_project starvla_mikasa \
  --wandb_entity "${WANDB_ENTITY}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix local/intercept_grab_fast_vla_v0_h1_train \
  --datasets.vla_data.eval_data_mix local/intercept_grab_fast_vla_v0_h1_val \
  --datasets.vla_data.custom_mixtures_path "${DATA_ROOT}/mikasa_intercept_grab_fast_h1.json" \
  --datasets.vla_data.per_device_batch_size "${MIKASA_MICRO_BATCH:-8}" \
  --trainer.gradient_accumulation_steps "${MIKASA_ACC_STEPS:-8}" \
  --trainer.max_train_steps 5000 \
  --trainer.profile_timing.enabled true \
  --trainer.pretrained_checkpoint "${INIT_CKPT}" \
  --checkpoint.save_training_state true \
  --checkpoint.save_pt_file true

SERVER_CHECKPOINT="${VLA_ROOT}/results/Checkpoints/${RUN_ID}/checkpoints/steps_5000_pytorch_model.pt"
"${VLA_PY}" deployment/model_server/server_policy.py \
  --ckpt_path "${SERVER_CHECKPOINT}" \
  --port "${VLA_PORT}" \
  --use_bf16 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}"' EXIT
sleep 5

cd "${REPO_ROOT}"
"${MIKASA_PY}" scripts/mikasa/evaluate.py \
  --policy vla \
  --host 127.0.0.1 \
  --port "${VLA_PORT}" \
  --episodes 50 \
  --output-dir "${TASK_ROOT}/vla_zero_delay_eval"
"${MIKASA_PY}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); n=sum(d["successes"]); print(f"{n}/50"); assert n >= 25' \
  "${TASK_ROOT}/vla_zero_delay_eval/summary.json"

printf 'MODEL_PATH=%s\n' "${VLA_ROOT}/results/Checkpoints/${RUN_ID}"
