#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_eval.sh [options]

Options:
  --run-dir <dir>          Training run directory (required)
  --gpus <csv>             CUDA visible devices, e.g. 0 or 1 (optional)
  --checkpoint <path>      Explicit checkpoint file (optional)
  --step <int>             Evaluate a specific step checkpoint (optional)
  --stage <name>           Output stage folder name (default: post_train)
  --config <path>          Explicit config path (default: <run-dir>/config.full.yaml)
  --workspace-dir <dir>    Workspace for resolving paths.* in experiment YAMLs
  --base-model-dir <dir>   Explicit local base VLM directory
  --base-model-repo-id <repo> Explicit HF base VLM repo id fallback
  --latencies <range>      Latencies for all evaluated tasks, e.g. 0-7 or 0,1,2
  --task-latencies <task=range>  Per-task latencies, repeatable, e.g. flappy=0-7
  --num-episodes <int>     Episodes per latency for all evaluated tasks
  --max-steps-per-episode <int>  Max env steps per episode for all evaluated tasks
  --task-num-episodes <task=int> Per-task episodes, repeatable
  --task-max-steps-per-episode <task=int> Per-task max steps, repeatable
  --override <key=value>   Raw OmegaConf override, repeatable
  --print-plan-only        Print resolved eval plan without loading checkpoint
  --wandb-enabled <true|false>  Enable wandb logging from eval script (default: true)
  --wandb-project <name>   Wandb project override (optional)
  --wandb-entity <name>    Wandb entity override (optional)
  --wandb-run-name <name>  Wandb run name override (optional)
  --help                   Show this help
USAGE
}

RUN_DIR=""
GPUS=""
CHECKPOINT=""
STEP=""
STAGE="post_train"
CONFIG=""
WORKSPACE_DIR_ARG=""
BASE_MODEL_DIR=""
BASE_MODEL_REPO_ID=""
LATENCIES=""
NUM_EPISODES=""
MAX_STEPS_PER_EPISODE=""
WANDB_ENABLED="true"
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_RUN_NAME=""
PRINT_PLAN_ONLY="false"
TASK_LATENCIES=()
TASK_NUM_EPISODES=()
TASK_MAX_STEPS_PER_EPISODE=()
OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --workspace-dir) WORKSPACE_DIR_ARG="$2"; shift 2 ;;
    --base-model-dir) BASE_MODEL_DIR="$2"; shift 2 ;;
    --base-model-repo-id) BASE_MODEL_REPO_ID="$2"; shift 2 ;;
    --latencies) LATENCIES="$2"; shift 2 ;;
    --task-latencies) TASK_LATENCIES+=("$2"); shift 2 ;;
    --num-episodes) NUM_EPISODES="$2"; shift 2 ;;
    --max-steps-per-episode) MAX_STEPS_PER_EPISODE="$2"; shift 2 ;;
    --task-num-episodes) TASK_NUM_EPISODES+=("$2"); shift 2 ;;
    --task-max-steps-per-episode) TASK_MAX_STEPS_PER_EPISODE+=("$2"); shift 2 ;;
    --override) OVERRIDES+=("$2"); shift 2 ;;
    --print-plan-only) PRINT_PLAN_ONLY="true"; shift ;;
    --wandb-enabled) WANDB_ENABLED="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "--run-dir is required"
  usage
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

CMD=(
  python examples/rl_games/scripts/eval_checkpoint.py
  --run-dir "$RUN_DIR"
  --stage "$STAGE"
  --wandb-enabled "$WANDB_ENABLED"
)
if [[ -n "$CHECKPOINT" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$STEP" ]]; then
  CMD+=(--step "$STEP")
fi
if [[ -n "$CONFIG" ]]; then
  CMD+=(--config "$CONFIG")
fi
if [[ -n "$WORKSPACE_DIR_ARG" ]]; then
  CMD+=(--workspace-dir "$WORKSPACE_DIR_ARG")
fi
if [[ -n "$BASE_MODEL_DIR" ]]; then
  CMD+=(--base-model-dir "$BASE_MODEL_DIR")
fi
if [[ -n "$BASE_MODEL_REPO_ID" ]]; then
  CMD+=(--base-model-repo-id "$BASE_MODEL_REPO_ID")
fi
if [[ -n "$LATENCIES" ]]; then
  CMD+=(--latencies "$LATENCIES")
fi
if [[ -n "$NUM_EPISODES" ]]; then
  CMD+=(--num-episodes "$NUM_EPISODES")
fi
if [[ -n "$MAX_STEPS_PER_EPISODE" ]]; then
  CMD+=(--max-steps-per-episode "$MAX_STEPS_PER_EPISODE")
fi
if [[ ${#TASK_LATENCIES[@]} -gt 0 ]]; then
  for item in "${TASK_LATENCIES[@]}"; do
    CMD+=(--task-latencies "$item")
  done
fi
if [[ ${#TASK_NUM_EPISODES[@]} -gt 0 ]]; then
  for item in "${TASK_NUM_EPISODES[@]}"; do
    CMD+=(--task-num-episodes "$item")
  done
fi
if [[ ${#TASK_MAX_STEPS_PER_EPISODE[@]} -gt 0 ]]; then
  for item in "${TASK_MAX_STEPS_PER_EPISODE[@]}"; do
    CMD+=(--task-max-steps-per-episode "$item")
  done
fi
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
  for item in "${OVERRIDES[@]}"; do
    CMD+=(--override "$item")
  done
fi
if [[ "$PRINT_PLAN_ONLY" == "true" ]]; then
  CMD+=(--print-plan-only)
fi
if [[ -n "$WANDB_PROJECT" ]]; then
  CMD+=(--wandb-project "$WANDB_PROJECT")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  CMD+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_RUN_NAME" ]]; then
  CMD+=(--wandb-run-name "$WANDB_RUN_NAME")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
