#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STARVLA_ROOT="${REPO_ROOT}/third_party/starVLA"
CHECKPOINT_ROOT="${STARVLA_ROOT}/results/Checkpoints"
VLA_REPO="latency-sensitive-bench/memory"
cd "${REPO_ROOT}"

BASE_MODEL_REPO="Qwen/Qwen3-VL-4B-Instruct"
BASE_MODEL_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
BASE_MODEL_DIR="${STARVLA_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"

usage() {
  cat >&2 <<'EOF'
Usage: eval_vla.sh [flappy|demon_attack|deadly_corridor|all] [single_baseline|plain_multi|stitch|kv_memory_flex|all]
EOF
}

select_envs() {
  case "$1" in
    flappy|demon_attack|deadly_corridor) ENVS=("$1") ;;
    all) ENVS=(flappy demon_attack deadly_corridor) ;;
    *) usage; exit 2 ;;
  esac
}

select_methods() {
  case "$1" in
    single_baseline|plain_multi|stitch|kv_memory_flex) METHODS=("$1") ;;
    all) METHODS=(single_baseline plain_multi stitch kv_memory_flex) ;;
    *) usage; exit 2 ;;
  esac
}

download_run() {
  local run_id="$1"

  conda run --no-capture-output -n latency hf download \
    "${VLA_REPO}" \
    --include \
    "${run_id}/checkpoints/steps_*_model.safetensors" \
    "${run_id}/config.yaml" \
    "${run_id}/config.full.yaml" \
    "${run_id}/dataset_statistics.json" \
    "${run_id}/wandb/wandb/run-*/files/config.yaml" \
    --local-dir "${CHECKPOINT_ROOT}"
}

write_latency_prompt_map() {
  local env_name="$1"
  local latency_raw_frames="$2"
  local output_path="$3"

  conda run --no-capture-output -n latency python - \
    "${env_name}" \
    "${latency_raw_frames}" \
    "${output_path}" <<'PY'
import json
import sys
from pathlib import Path

from latency_bench.core.config import load_config
from latency_bench.data.prompts import render_teacher_prompt

env_name = sys.argv[1]
latency_raw_frames = int(sys.argv[2])
output_path = Path(sys.argv[3])
env = load_config(Path("configs/base") / f"{env_name}.yaml")["env"]
latency_ms = latency_raw_frames * 1000.0 / env["env_fps"]
prompt = render_teacher_prompt(
    env_name,
    latency_raw_frames=latency_raw_frames,
    latency_ms=latency_ms,
    env_fps=env["env_fps"],
    obs_fps=env["obs_fps"],
)
output_path.write_text(
    json.dumps(
        {
            str(latency_raw_frames): {
                "latency": latency_raw_frames,
                "latency_raw_frames": latency_raw_frames,
                "latency_ms": latency_ms,
                "prompt": prompt,
            }
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
}

latest_wandb_run_id() {
  local run_dir="$1"
  local config_path run_dir_name

  config_path="$(
    find "${run_dir}/wandb" -type f -path '*/wandb/run-*/files/config.yaml' \
      | sort \
      | tail -n 1
  )"
  if [[ -z "${config_path}" ]]; then
    echo "No WandB run metadata found under ${run_dir}/wandb" >&2
    return 1
  fi
  run_dir_name="$(basename "$(dirname "$(dirname "${config_path}")")")"
  echo "${run_dir_name##*-}"
}

eval_method() {
  local env_name="$1"
  local method="$2"
  local latency episodes eval_parallel raw_name run_id run_dir prompt_map wandb_run_id

  case "${env_name}" in
    flappy)
      latency=3
      episodes=200
      eval_parallel=20
      ;;
    demon_attack)
      latency=6
      episodes=200
      eval_parallel=20
      ;;
    deadly_corridor)
      latency=6
      episodes=1000
      eval_parallel=1
      ;;
  esac

  raw_name="${env_name}_fix_latency_${latency}_${episodes}ep_7k2steps"
  run_id="${raw_name}_${method}"
  run_dir="${CHECKPOINT_ROOT}/${run_id}"

  if ! compgen -G "${run_dir}/checkpoints/steps_*_model.safetensors" >/dev/null; then
    download_run "${run_id}"
  fi

  prompt_map="${run_dir}/eval_inputs/${env_name}_fixed_latency_${latency}.json"
  mkdir -p "$(dirname "${prompt_map}")"
  write_latency_prompt_map "${env_name}" "${latency}" "${prompt_map}"
  wandb_run_id="$(latest_wandb_run_id "${run_dir}")"

  echo "Evaluating ${run_id}: latency=${latency} raw_frames, episodes=100, parallel=${eval_parallel}, wandb_run_id=${wandb_run_id}"
  conda run --no-capture-output -n starvla_rl_games_openvla python \
    "${STARVLA_ROOT}/examples/rl_games/scripts/eval_checkpoint.py" \
    --run-dir "${run_dir}" \
    --stage post_train \
    --workspace-dir "${STARVLA_ROOT}" \
    --base-model-dir "${BASE_MODEL_DIR}" \
    --latencies "${latency}" \
    --num-episodes 100 \
    --override "rl_games.env_eval.enabled=true" \
    --override "rl_games.env_eval.eval_backend=latency_bench" \
    --override "rl_games.env_eval.eval_parallel_envs=${eval_parallel}" \
    --override "rl_games.env_eval.post_train.enabled=true" \
    --override "rl_games.env_eval.latency.prompt_map_path=${prompt_map}" \
    --wandb-enabled true \
    --wandb-run-id "${wandb_run_id}"
}

select_envs "${1:-all}"
select_methods "${2:-all}"

conda run --no-capture-output -n latency hf download \
  "${BASE_MODEL_REPO}" \
  --revision "${BASE_MODEL_REVISION}" \
  --local-dir "${BASE_MODEL_DIR}"

for env_name in "${ENVS[@]}"; do
  for method in "${METHODS[@]}"; do
    eval_method "${env_name}" "${method}"
  done
done
