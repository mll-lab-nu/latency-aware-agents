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
UV="${UV:?packaging uv executable}"
TASK_PY="$ROOT/third_party/MIKASA-Robo/.venv/bin/python"
DATA="$ROOT/outputs/starvla/datasets/mikasa_intercept_grab_fast/$RUN_ID/$CONDITION"
SFT="$ROOT/outputs/starvla/runs/mikasa_intercept_grab_fast/$RUN_ID/sft"
DAGGER="$ROOT/outputs/starvla/runs/mikasa_intercept_grab_fast/$RUN_ID/dagger"
STATE="$ROOT/outputs/tasks/mikasa_intercept_grab_fast/$RUN_ID"
TEACHER="$ROOT/outputs/starvla/artifacts/models/zero-latency/mikasa-intercept-grab-fast/small-policy/ppo-v1/checkpoint.pt"
cd "$ROOT"
set -a
. ./.env
set +a
unset WANDB_RUN_ID WANDB_RESUME
export WANDB_ENTITY=zihanwang-ai-northwestern-university
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/third_party/starVLA"
export CUDA_VISIBLE_DEVICES="$DEVICES"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="$(dirname "$(dirname "$VLA_PY")")"
export PATH="$CUDA_HOME/bin:$PATH"
PUBLISH_PREFIX="latency-aware/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h8/$RUN_ID"
mkdir -p "$DATA" "$STATE"
publish_failure() {
  job_exit=$?
  trap - ERR
  printf 'failed:%s\n' "$job_exit" > "$STATE/phase"
  "$VLA_PY" - "$SFT" "$DAGGER" "$STATE" "$PUBLISH_PREFIX" <<'PY'
from pathlib import Path
import sys
from scripts.tasks.mikasa_intercept_grab_fast.publish_prefix63 import publish_diagnostics
publish_diagnostics(*map(Path, sys.argv[1:4]), sys.argv[4])
PY
  exit "$job_exit"
}
trap publish_failure ERR

"$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.write_conditioned_latency_config \
  --condition "$CONDITION" --output "$STATE/$CONDITION.yaml"
printf 'collect-teacher-prefix63\n' > "$STATE/phase"
if [[ ! -f "$DATA/raw/collection.json" ]]; then
  "$TASK_PY" -m scripts.tasks.mikasa_intercept_grab_fast.collect_conditioned_h8 \
    --condition "$CONDITION" --model shared --latency-config "$STATE/$CONDITION.yaml" \
    --checkpoint "$TEACHER" --output "$DATA/raw" --train-targets 64288 \
    --validation-targets 6884 --num-envs 128 > "$STATE/collection.log" 2>&1
fi
printf 'package-prefix63\n' > "$STATE/phase"
if [[ ! -f "$DATA/lerobot/task_contract.json" ]]; then
  "$UV" run --frozen --project "$ROOT/third_party/MIKASA-Robo/utils/convert_rlds_to_lerobot" python \
    -m scripts.tasks.mikasa_intercept_grab_fast.prepare_conditioned_h8_dataset \
    --raw-root "$DATA/raw" --output-train "$DATA/lerobot" > "$STATE/package.log" 2>&1
fi
"$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.audit_conditioned_h8_dataset \
  --condition "$CONDITION" --raw-root "$DATA/raw" --dataset "$DATA/lerobot" > "$STATE/data-audit.log" 2>&1
printf 'sft-5000\n' > "$STATE/phase"
if [[ ! -f "$SFT/inference/provenance.json" ]]; then
  RESUME=()
  if [[ -f "$SFT/run.json" ]]; then RESUME+=(--resume); fi
  "$VLA_PY" -m scripts.starvla.train_task --task mikasa_conditioned_h8 --model qwenoft \
    --condition "$CONDITION" --train-only --no-oom-retry --no-gradient-checkpointing \
    --run-id "$RUN_ID" --output-dir "$SFT" --dataset-dir "$DATA" \
    --no-publish --vla-python "$VLA_PY" \
    --num-gpus 2 --micro-batch 64 --port "$SFT_PORT" --task-python "$TASK_PY" \
    "${RESUME[@]}" > "$STATE/sft.log" 2>&1
fi
printf 'publish-sft-endpoint\n' > "$STATE/phase"
"$VLA_PY" - "$SFT/training/bootstrap" "$PUBLISH_PREFIX" <<'PY'
from pathlib import Path
import sys
from scripts.tasks.mikasa_intercept_grab_fast.publish_prefix63 import publish_training_endpoint
publish_training_endpoint(Path(sys.argv[1]), sys.argv[2])
PY
"$VLA_PY" - "$SFT" "$STATE" "$VLA_PY" "$CONDITION" <<'PY'
from pathlib import Path
from types import SimpleNamespace
import sys
import yaml
from scripts.starvla.train_task import eval_config
from scripts.tasks.mikasa_intercept_grab_fast.fixed_latency import FIXED_LATENCIES
sft, state, python = map(Path, sys.argv[1:4])
condition = sys.argv[4]
training = yaml.safe_load((sft / 'bootstrap.yaml').read_text())
args = SimpleNamespace(task='mikasa_conditioned_h8', model='qwenoft', condition=condition,
    eval_seed=4242424442, inference_device='cuda:1', simulator_device='cuda:0', smoke=False,
    vla_python=python, backbone_path=Path(training['framework']['qwenvl']['base_vlm']))
config = eval_config(args, sft / 'inference', Path('unused'), state / 'development',
                     eval_episodes=100, eval_latency_frames=FIXED_LATENCIES[condition])
(state / 'eval.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
PY
printf 'dagger-5000-five-latencies\n' > "$STATE/phase"
"$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.run_dagger \
  --model qwenoft --condition "$CONDITION" --source-run "$SFT" --bootstrap "$DATA" \
  --eval-config "$STATE/eval.yaml" \
  --latency-config "$STATE/$CONDITION.yaml" --teacher-checkpoint "$TEACHER" --output "$DAGGER" \
  --vla-python "$VLA_PY" --task-python "$TASK_PY" --uv "$UV" \
  --train-devices "$DEVICES" --inference-gpu 1 --simulator-gpu 0 \
  --gpu-lock "/tmp/mikasa-h8-dagger-gpu${DEVICES%%,*}.lock" --port "$DAGGER_PORT" > "$STATE/dagger.log" 2>&1
printf 'report\n' > "$STATE/phase"
REPORT_ARGS=()
if [[ -n "$PREVIOUS_COLUMN" ]]; then REPORT_ARGS+=(--previous-column "$PREVIOUS_COLUMN"); fi
"$VLA_PY" -m scripts.tasks.mikasa_intercept_grab_fast.prefix_repair_results \
  --run "$DAGGER" --output "$STATE/results" "${REPORT_ARGS[@]}"
"$VLA_PY" - "$DAGGER" "$STATE" "$PUBLISH_PREFIX" <<'PY'
from pathlib import Path
import json,sys
from scripts.tasks.mikasa_intercept_grab_fast.publish_prefix63 import publish_folder, publish_training_endpoint, MODEL_REPO
dagger,state,prefix=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
publish_training_endpoint(dagger/'training/dagger', prefix)
best=json.loads((dagger/'best.json').read_text())
bundle=Path(best['checkpoint']).parent.parent
publish_folder(bundle,prefix,repo_id=MODEL_REPO,repo_type='model')
publish_folder(state/'results',prefix+'/evaluation',repo_id=MODEL_REPO,repo_type='model')
PY
printf 'complete\n' > "$STATE/phase"
