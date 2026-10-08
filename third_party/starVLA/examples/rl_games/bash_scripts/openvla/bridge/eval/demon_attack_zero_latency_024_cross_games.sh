#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://huggingface.co

# Eval-only sweep for the Demon Attack fixed-0-latency OpenVLA bridge policy.
# Evaluates latency labels, not raw frame skips, on Demon Attack, Asterix,
# Atlantis, and Air Raid.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${REPO_ROOT}/.." && pwd)}"
export WORKSPACE_DIR

MODEL_TAG="demon_attack_zero_latency"
HF_REPO_ID="${HF_REPO_ID:-latency-sensitive-bench/benchmark-models}"
HF_REVISION="${HF_REVISION:-671e2a7451fda7f7dba5606c6d11183ebd804b6d}"
HF_SUBDIR="${HF_SUBDIR:-zero-latency/demon-attack/vla/starvla-qwenoft-h1/demon_attack_fix_latency_0_200ep}"
CHECKPOINT_REL="${CHECKPOINT_REL:-checkpoints/steps_7000_model.safetensors}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-7000}"

TASKS="${TASKS:-demon_attack,asterix,atlantis,air_raid}"
LATENCIES="${LATENCIES:-0,2,4}"
NUM_EPISODES="${NUM_EPISODES:-20}"
MAX_STEPS_PER_EPISODE="${MAX_STEPS_PER_EPISODE:-3600}"
GPUS="${GPUS:-}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-${MODEL_TAG}_latency024}"
PRINT_PLAN_ONLY="${PRINT_PLAN_ONLY:-0}"
RUN_INSTALL="${RUN_INSTALL:-0}"
RUN_LATENCY_DEPS="${RUN_LATENCY_DEPS:-0}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-1}"
CONDA_ENV="${CONDA_ENV:-starvla_rl_games_openvla}"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-${REPO_ROOT}/results/HFCheckpoints}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/results/Evals/${MODEL_TAG}_latency024_cross_games}"

cd "${REPO_ROOT}"

detect_latency_bench_root() {
  if [[ -n "${LATENCY_BENCH_ROOT:-}" && -d "${LATENCY_BENCH_ROOT}" ]]; then
    return
  fi
  if [[ -d "${WORKSPACE_DIR}/latency-sensitive-bench" ]]; then
    LATENCY_BENCH_ROOT="${WORKSPACE_DIR}/latency-sensitive-bench"
  elif [[ -d "${WORKSPACE_DIR}/latency-sensitive-bench-realtime-human" ]]; then
    LATENCY_BENCH_ROOT="${WORKSPACE_DIR}/latency-sensitive-bench-realtime-human"
  else
    LATENCY_BENCH_ROOT=""
  fi
  export LATENCY_BENCH_ROOT
}

activate_conda_env() {
  if [[ "${ACTIVATE_CONDA}" != "1" ]]; then
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda/etc/profile.d/conda.sh"
  else
    echo "[error] conda not found; set ACTIVATE_CONDA=0 if your shell is already activated" >&2
    exit 2
  fi
  conda activate "${CONDA_ENV}"
}

download_hf_snapshot() {
  local repo_id="$1"
  local revision="$2"
  local local_dir="$3"
  shift 3
  python - "$repo_id" "$revision" "$local_dir" "$@" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id, revision, local_dir, *patterns = sys.argv[1:]
Path(local_dir).mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    revision=revision or None,
    local_dir=local_dir,
    allow_patterns=patterns or None,
)
PY
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[error] expected file is missing: ${path}" >&2
    exit 2
  fi
}

is_print_plan_only() {
  case "${PRINT_PLAN_ONLY}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

add_common_eval_args() {
  EVAL_ARGS=(
    --step "${CHECKPOINT_STEP}"
    --stage post_train
    --latencies "${LATENCIES}"
    --num-episodes "${NUM_EPISODES}"
    --max-steps-per-episode "${MAX_STEPS_PER_EPISODE}"
    --wandb-enabled "${WANDB_ENABLED}"
    --workspace-dir "${WORKSPACE_DIR}"
    --base-model-repo-id "Qwen/Qwen3-VL-4B-Instruct"
    --override "rl_games.env_eval.eval_backend=latency_bench"
    --override "rl_games.env_eval.mid_train.enabled=false"
    --override "rl_games.env_eval.post_train.enabled=true"
    --override "rl_games.env_eval.frameskip=4"
    --override "rl_games.env_eval.latency.prompt_map_path=null"
  )
  if [[ -n "${GPUS}" ]]; then
    EVAL_ARGS+=(--gpus "${GPUS}")
  fi
  if [[ -n "${WANDB_PROJECT}" ]]; then
    EVAL_ARGS+=(--wandb-project "${WANDB_PROJECT}")
  fi
  if [[ -n "${WANDB_ENTITY}" ]]; then
    EVAL_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if is_print_plan_only; then
    EVAL_ARGS+=(--print-plan-only)
  fi
}

add_task_overrides() {
  local task="$1"
  TASK_OVERRIDES=(
    --override "env=${task}"
    --override "rl_games.task=${task}"
  )
  case "${task}" in
    demon_attack)
      TASK_OVERRIDES+=(
        --override "rl_games.env_eval.task_description='You are playing Demon Attack from a single game image. Choose exactly one action from: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE.'"
        --override "rl_games.env_eval.demon_attack.noop_max=30"
      )
      ;;
    asterix)
      TASK_OVERRIDES+=(
        --override "rl_games.env_eval.task_description='Collect useful objects and avoid lyres. Choose exactly one action from: noop, up, right, left, down, upright, upleft, downright, downleft.'"
        --override "rl_games.env_eval.asterix.action_layout=factorized_6"
        --override "rl_games.env_eval.atari.env_id=ALE/Asterix-v5"
        --override "rl_games.env_eval.atari.gym_frameskip=4"
        --override "rl_games.env_eval.atari.noop_max=30"
        --override "rl_games.env_eval.atari.repeat_action_probability=0.0"
        --override "rl_games.env_eval.atari.full_action_space=false"
      )
      ;;
    atlantis)
      TASK_OVERRIDES+=(
        --override "rl_games.env_eval.task_description='Defend Atlantis by firing at descending enemies. Choose exactly one action from: noop, fire, rightfire, leftfire.'"
        --override "rl_games.env_eval.atari.env_id=ALE/Atlantis-v5"
        --override "rl_games.env_eval.atari.gym_frameskip=4"
        --override "rl_games.env_eval.atari.noop_max=30"
        --override "rl_games.env_eval.atari.repeat_action_probability=0.0"
        --override "rl_games.env_eval.atari.full_action_space=false"
      )
      ;;
    air_raid)
      TASK_OVERRIDES+=(
        --override "rl_games.env_eval.task_description='Protect both buildings from flying saucers. Choose exactly one action from: noop, fire, right, left, rightfire, leftfire.'"
        --override "rl_games.env_eval.atari.env_id=ALE/AirRaid-v5"
        --override "rl_games.env_eval.atari.gym_frameskip=1"
        --override "rl_games.env_eval.atari.noop_max=30"
        --override "rl_games.env_eval.atari.fire_reset=true"
        --override "rl_games.env_eval.atari.mode=1"
        --override "rl_games.env_eval.atari.difficulty=0"
        --override "rl_games.env_eval.atari.repeat_action_probability=0.0"
        --override "rl_games.env_eval.atari.full_action_space=false"
        --override "rl_games.env_eval.atari.max_num_frames_per_episode=108000"
      )
      ;;
    *)
      echo "[error] unsupported task '${task}'" >&2
      exit 2
      ;;
  esac
}

detect_latency_bench_root

if [[ "${RUN_INSTALL}" == "1" ]]; then
  bash examples/rl_games/install/install_stack.sh openvla demon_attack
fi

activate_conda_env

if [[ -n "${LATENCY_BENCH_ROOT}" ]]; then
  export PYTHONPATH="${LATENCY_BENCH_ROOT}:${PYTHONPATH:-}"
fi

if [[ "${RUN_LATENCY_DEPS}" == "1" ]]; then
  if [[ -d "${WORKSPACE_DIR}/latency-sensitive-bench/third_party/flappy-bird-gymnasium" ]]; then
    bash examples/rl_games/bash_scripts/install/latency_deps.sh
  else
    echo "[warn] skipping latency_deps.sh because ${WORKSPACE_DIR}/latency-sensitive-bench is not present" >&2
  fi
fi

HF_LOCAL_DIR="${HF_CACHE_ROOT}/openvla_demon_attack_200ep"
HF_ALLOW_PATTERNS=(
  "${HF_SUBDIR}/config.yaml"
  "${HF_SUBDIR}/config.full.yaml"
  "${HF_SUBDIR}/dataset_statistics.json"
  "${HF_SUBDIR}/dataset_statistics_eval.json"
  "${HF_SUBDIR}/hydra/.hydra/config.yaml"
  "${HF_SUBDIR}/hydra/.hydra/hydra.yaml"
  "${HF_SUBDIR}/hydra/.hydra/overrides.yaml"
)
if ! is_print_plan_only; then
  HF_ALLOW_PATTERNS+=("${HF_SUBDIR}/${CHECKPOINT_REL}")
fi
download_hf_snapshot \
  "${HF_REPO_ID}" \
  "${HF_REVISION}" \
  "${HF_LOCAL_DIR}" \
  "${HF_ALLOW_PATTERNS[@]}"

CONFIG_PATH="${HF_LOCAL_DIR}/${HF_SUBDIR}/config.full.yaml"
CHECKPOINT_PATH="${HF_LOCAL_DIR}/${HF_SUBDIR}/${CHECKPOINT_REL}"
require_file "${CONFIG_PATH}"
if ! is_print_plan_only; then
  require_file "${CHECKPOINT_PATH}"
fi

TASKS_NORMALIZED="${TASKS//,/ }"
read -r -a TASK_ARRAY <<< "${TASKS_NORMALIZED}"
add_common_eval_args

for raw_task in "${TASK_ARRAY[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  [[ -n "${task}" ]] || continue
  add_task_overrides "${task}"
  TASK_RUN_DIR="${RUN_ROOT}/${task}"
  mkdir -p "${TASK_RUN_DIR}"

  CMD=(
    bash examples/rl_games/scripts/run_eval.sh
    --run-dir "${TASK_RUN_DIR}"
    --config "${CONFIG_PATH}"
    --checkpoint "${CHECKPOINT_PATH}"
    --wandb-run-name "${WANDB_RUN_PREFIX}_${task}"
    "${EVAL_ARGS[@]}"
    "${TASK_OVERRIDES[@]}"
  )

  printf '[eval:%s]' "${task}"
  printf ' %q' "${CMD[@]}"
  echo
  "${CMD[@]}"
done
