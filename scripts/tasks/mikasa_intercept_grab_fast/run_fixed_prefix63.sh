#!/usr/bin/env bash
# Formal Fixed-N prefix63 training on an explicitly allocated GPU pair.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDITION="${1:?fixed-N condition}"
RUN_ID="${2:?unique run ID}"
DEVICES="${3:?physical GPU pair}"
SFT_PORT="${4:?SFT DDP port}"
DAGGER_PORT="${5:?DAgger DDP port}"
PREVIOUS_COLUMN="${6-}" # Optional old five-cell column for the Fixed-2 repair comparison.
VLA_PY="${VLA_PY:?QwenOFT Python}"
LEROBOT_PY="${LEROBOT_PY:?installed LeRobot interpreter}"
TASK_PY="${TASK_PY:-$(command -v python)}"
DATA="$ROOT/outputs/starvla/datasets/mikasa_intercept_grab_fast/$RUN_ID/$CONDITION"
SFT="$ROOT/outputs/starvla/runs/mikasa_intercept_grab_fast/$RUN_ID/sft"
DAGGER="$ROOT/outputs/starvla/runs/mikasa_intercept_grab_fast/$RUN_ID/dagger"
STATE="$ROOT/outputs/tasks/mikasa_intercept_grab_fast/$RUN_ID"
TEACHER="$ROOT/outputs/starvla/artifacts/models/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt"
cd "$ROOT"
unset WANDB_RUN_ID WANDB_RESUME
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/third_party/starVLA"
export CUDA_VISIBLE_DEVICES="$DEVICES"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="$(dirname "$(dirname "$VLA_PY")")"
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$DATA" "$STATE"
record_failure() {
  job_exit=$?
  trap - ERR
  printf 'failed:%s\n' "$job_exit" > "$STATE/phase"
  exit "$job_exit"
}
trap record_failure ERR

"$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.write_conditioned_configs \
  --condition "$CONDITION" --output "$STATE/$CONDITION.yaml"
export TASK_PY LEROBOT_PY VLA_PY
export TEACHER_CHECKPOINT="$TEACHER" DATASET_DIR="$DATA" OUTPUT_DIR="$SFT"
printf 'collect-teacher-prefix63\n' > "$STATE/phase"
bash scripts/tasks/mikasa_intercept_grab_fast/run_conditioned_h8.sh \
  collect qwenoft "$CONDITION" "$RUN_ID" "$DEVICES" "$SFT_PORT" > "$STATE/collection.log" 2>&1
printf 'sft-5000\n' > "$STATE/phase"
bash scripts/tasks/mikasa_intercept_grab_fast/run_conditioned_h8.sh \
  train qwenoft "$CONDITION" "$RUN_ID" "$DEVICES" "$SFT_PORT" > "$STATE/sft.log" 2>&1
printf 'prepare-dagger-evaluation\n' > "$STATE/phase"
"$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.write_conditioned_configs \
  --condition "$CONDITION" --output "$STATE/eval.yaml" \
  --evaluation-run "$SFT" --vla-python "$VLA_PY" \
  --evaluation-config configs/examples/mikasa_intercept_grab_fast/fixed_prefix63_eval.yaml
printf 'dagger-5000-five-latencies\n' > "$STATE/phase"
"$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.run_dagger \
  --model qwenoft --condition "$CONDITION" --source-run "$SFT" --bootstrap "$DATA" \
  --eval-config "$STATE/eval.yaml" \
  --latency-config "$STATE/$CONDITION.yaml" --teacher-checkpoint "$TEACHER" --output "$DAGGER" \
  --vla-python "$VLA_PY" --task-python "$TASK_PY" --lerobot-python "$LEROBOT_PY" \
  --train-devices "$DEVICES" --inference-gpu 1 --simulator-gpu 0 \
  --gpu-lock "/tmp/mikasa-h8-dagger-gpu${DEVICES%%,*}.lock" --port "$DAGGER_PORT" > "$STATE/dagger.log" 2>&1
printf 'report\n' > "$STATE/phase"
REPORT_ARGS=()
if [[ -n "$PREVIOUS_COLUMN" ]]; then REPORT_ARGS+=(--previous-column "$PREVIOUS_COLUMN"); fi
"$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.prefix_repair_results \
  --run "$DAGGER" --output "$STATE/results" "${REPORT_ARGS[@]}"
printf 'complete\n' > "$STATE/phase"
