#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/checkpoints/demon_attack_ppo_fs0 logs

GPU_ID="${GPU_ID:-0}"
N_ENVS="${N_ENVS:-16}"
N_STEPS="${N_STEPS:-128}"
TRAIN_STEPS="${TRAIN_STEPS:-20000000}"
EXP_NAME="${EXP_NAME:-demon_attack_ppo_fs0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$(hostname -s)_$$}"
RUN_DIR="${RUN_DIR:-logs/${EXP_NAME}/${RUN_ID}}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/train.log}"
METRICS_PATH="${METRICS_PATH-${RUN_DIR}/metrics.jsonl}"

mkdir -p "${RUN_DIR}" "$(dirname "${LOG_PATH}")"

export PYTHONUNBUFFERED=1
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

printf '%q ' "$0" "$@" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m training.demon_attack_ppo.cli \
    --env-id ALE/DemonAttack-v5 \
    --train-steps "${TRAIN_STEPS}" \
    --n-envs "${N_ENVS}" \
    --n-steps "${N_STEPS}" \
    --batch-size 256 \
    --n-epochs 8 \
    --learning-rate 2.5e-4 \
    --learning-rate-final 2.5e-6 \
    --anneal-lr \
    --clip-coef 0.1 \
    --clip-coef-final 0.1 \
    --ent-coef 0.01 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --target-kl 0.015 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --screen-size 84 \
    --base-channels 1 \
    --stack-frames 4 \
    --noop-max 30 \
    --action-repeat "${ACTION_REPEAT:-4}" \
    --terminal-on-life-loss \
    --clip-reward \
    --obs-mode image_only \
    --frame-skip-values 0 \
    --checkpoint-freq 500000 \
    --save results/checkpoints/demon_attack_ppo_fs0/demon_attack_ppo_fs0.pt \
    --device cuda \
    --log-interval-seconds 10 \
    --wandb-project latency \
    --wandb-run-name demon-attack-ppo-fs0 \
    --run-id "${RUN_ID}" \
    --metrics-jsonl "${METRICS_PATH}" \
    2>&1 | tee "${LOG_PATH}"

case "${LOG_PATH}" in
    /*) LATEST_TARGET="${LOG_PATH}" ;;
    *) LATEST_TARGET="${PWD}/${LOG_PATH}" ;;
esac
ln -sfn "${LATEST_TARGET}" "logs/${EXP_NAME}_latest.log" || true
printf '%s\n' "${LOG_PATH}" > "logs/${EXP_NAME}_latest.path"
