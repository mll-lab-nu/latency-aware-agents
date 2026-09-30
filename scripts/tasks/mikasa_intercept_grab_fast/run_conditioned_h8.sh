#!/usr/bin/env bash
# Collect, train, and evaluate one MIKASA conditioned H8 experiment.
set -euo pipefail
PHASE="${1:?collect|train|eval|all}"
MODEL="${2:?model}"
CONDITION="${3:?profile-latency|fixed-N}"
RUN_ID="${4:?run ID}"
DEVICES="${5:?physical GPU pair}"
PORT="${6:?training port}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VLA_PY="${VLA_PY:?model interpreter}"
TASK_PY="${TASK_PY:?simulator interpreter}"
UV="${UV:?packaging uv executable}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:?teacher checkpoint}"
DATASET_DIR="${DATASET_DIR:?dataset directory}"
OUTPUT_DIR="${OUTPUT_DIR:?training directory}"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$DEVICES"
export PYTHONPATH="$ROOT:$ROOT/third_party/MIKASA-Robo:$ROOT/third_party/starVLA"

collect() {
  mkdir -p "$DATASET_DIR"
  "$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.write_conditioned_latency_config \
    --condition "$CONDITION" --model "$MODEL" --output "$DATASET_DIR/latency.yaml"
  local collection_model="$MODEL"
  if [[ "$CONDITION" == fixed-* ]]; then collection_model=shared; fi
  if [[ ! -f "$DATASET_DIR/raw/collection.json" ]]; then
    "$TASK_PY" -m scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 \
      --condition "$CONDITION" --model "$collection_model" --latency-config "$DATASET_DIR/latency.yaml" \
      --checkpoint "$TEACHER_CHECKPOINT" --output "$DATASET_DIR/raw" \
      --train-targets 64288 --validation-targets 6884 --num-envs 128
  fi
  if [[ ! -f "$DATASET_DIR/lerobot/task_contract.json" ]]; then
    "$UV" run --frozen --project "$ROOT/third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot" python \
      -m scripts.tasks.mikasa_intercept_grab_fast.prepare_conditioned_h8_dataset \
      --raw-root "$DATASET_DIR/raw" --output-train "$DATASET_DIR/lerobot"
  fi
  "$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.audit_conditioned_h8_dataset \
    --condition "$CONDITION" --raw-root "$DATASET_DIR/raw" --dataset "$DATASET_DIR/lerobot"
}

train() {
  local resume=()
  if [[ -f "$OUTPUT_DIR/run.json" ]]; then resume+=(--resume); fi
  local checkpointing=--gradient-checkpointing
  if [[ "$MODEL" == qwenoft ]]; then checkpointing=--no-gradient-checkpointing; fi
  "$VLA_PY" -m scripts.starvla.train_task --task mikasa_conditioned_h8 --model "$MODEL" \
    --condition "$CONDITION" --train-only --no-oom-retry "$checkpointing" \
    --run-id "$RUN_ID" --output-dir "$OUTPUT_DIR" --dataset-dir "$DATASET_DIR" \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --no-publish --vla-python "$VLA_PY" --task-python "$TASK_PY" \
    --num-gpus 2 --micro-batch 64 --port "$PORT" "${resume[@]}"
}

evaluate() {
  local setting="$CONDITION"
  if [[ "$CONDITION" == profile-latency ]]; then setting=profile; fi
  "$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.evaluate_h8_checkpoint \
    --model "$MODEL" --run-dir "$OUTPUT_DIR" --vla-python "$VLA_PY" --task-python "$TASK_PY" \
    --eval-cuda-visible-devices "$DEVICES" --setting "$setting" \
    --inference-device cuda:1 --simulator-device cuda:0 \
    --seed 4242424242 --episodes 200 --label "final-$setting" --skip-wandb
}

case "$PHASE" in
  collect) collect ;;
  train) train ;;
  eval) evaluate ;;
  all) collect; train; evaluate ;;
  *) echo "Unknown phase: $PHASE" >&2; exit 2 ;;
esac
