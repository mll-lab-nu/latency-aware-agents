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
  "$TASK_PY" - "$DATASET_DIR" "$TEACHER_CHECKPOINT" "$CONDITION" "$collection_model" <<'PYCFG'
import sys
from pathlib import Path
import yaml
from latency_bench.core.config import load_config
root, checkpoint = map(Path, sys.argv[1:3])
condition, model = sys.argv[3:5]
config = load_config(root / "latency.yaml")
prefix = ({"mode": "profile_p95", "profile_path": config["latency"]["profile_path"],
           "profile_revision": config["profile_source"]["revision"]} if condition == "profile-latency"
          else {"mode": "fixed_frames", "frames": int(condition.split("-")[1])})
config.update(experiment={"name": "mikasa_h8_collection", "seed": 0},
    policy={"type": "mikasa_ppo", "source": "local", "root": str(checkpoint.resolve().parent),
            "checkpoint": checkpoint.name, "action_prefix": prefix},
    evaluation={"max_env_steps": 60}, logging={"output_dir": str(root / "raw")},
    collection={"label_mode": "prefix_h8", "condition": condition, "model": model,
                "train_targets": 64288, "validation_targets": 6884})
config["env"].update(name="mikasa_intercept_grab_fast", env_fps=20, obs_fps=20, simulator_device="gpu")
config["executor"].update(mode="simulated", parallel_envs=128, inference_devices=["cuda:0"])
(root / "collect.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
(root / "convert.yaml").write_text(yaml.safe_dump({"conversion": {
    "format": "lerobot", "input_dir": str(root / "raw"), "output_dir": str(root / "lerobot")}}))
PYCFG
  if [[ ! -f "$DATASET_DIR/raw/collection.json" ]]; then
    "$TASK_PY" -m latency_bench collect --config "$DATASET_DIR/collect.yaml"
  fi
  if [[ ! -f "$DATASET_DIR/lerobot/task_contract.json" ]]; then
    "$UV" run --frozen --project "$ROOT/third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot" python \
      -m latency_bench convert --config "$DATASET_DIR/convert.yaml"
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
