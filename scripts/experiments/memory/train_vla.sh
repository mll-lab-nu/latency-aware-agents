#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STARVLA_ROOT="${REPO_ROOT}/third_party/starVLA"
RAW_ROOT="${REPO_ROOT}/teacher_datasets/memory"
VLA_REPO="latency-sensitive-bench/memory"
cd "${REPO_ROOT}"

BASE_MODEL_REPO="Qwen/Qwen3-VL-4B-Instruct"
BASE_MODEL_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
BASE_MODEL_DIR="${STARVLA_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
INITIALIZER_REPO="StarVLA/Qwen3VL-OFT-Bridge-RT-1"
INITIALIZER_REVISION="c3fc8f028429ba14819bf3b16e098776b670c889"
INITIALIZER_CHECKPOINT="checkpoints/steps_5000_pytorch_model.pt"
INITIALIZER_DIR="${STARVLA_ROOT}/playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1"

usage() {
  cat >&2 <<'EOF'
Usage: train_vla.sh [flappy|demon_attack|deadly_corridor|all] [single_baseline|plain_multi|stitch|kv_memory_flex|all]
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

prepare_data() {
  local env_name="$1"
  local latency episodes raw_name data_root data_mix converter
  local converter_args=()

  case "${env_name}" in
    flappy)
      latency=3
      episodes=200
      converter="${STARVLA_ROOT}/examples/rl_games/bash_scripts/gr00t/data_conversion/convert_flappy_to_starvla_lerobot.py"
      ;;
    demon_attack)
      latency=6
      episodes=200
      converter="${STARVLA_ROOT}/examples/rl_games/bash_scripts/gr00t/data_conversion/convert_demon_attack_to_starvla_lerobot.py"
      ;;
    deadly_corridor)
      latency=6
      episodes=1000
      converter="${STARVLA_ROOT}/examples/rl_games/bash_scripts/gr00t/data_conversion/convert_deadly_corridor_to_starvla_lerobot.py"
      converter_args=(--action-layout multibinary_7)
      ;;
  esac

  raw_name="${env_name}_fix_latency_${latency}_${episodes}ep_7k2steps"
  data_root="${STARVLA_ROOT}/data/memory/${raw_name}"
  data_mix="${env_name}_train__bridge"

  conda run --no-capture-output -n starvla_rl_games_openvla python \
    "${converter}" \
    --dataset-name "${RAW_ROOT}/${raw_name}" \
    --source-metadata "${RAW_ROOT}/${raw_name}/metadata.json" \
    --source-latency-column latency_raw_frames \
    --target-latency-unit raw_frames \
    --output-dir "${data_root}/${data_mix}" \
    --action-carrier bridge \
    --force \
    "${converter_args[@]}"
}

train_method() {
  local env_name="$1"
  local method="$2"
  local latency episodes raw_name run_id dataset_local_dir command run_dir
  local eval_args=()

  # RL-games online latency is measured in decision steps. Raw fixed-6 is not an
  # integer decision delay for the stride-4 Atari/Doom environments.
  case "${env_name}" in
    flappy)
      latency=3
      episodes=200
      eval_args=(
        "rl_games.env_eval.mid_train.latencies=[3]"
        "rl_games.env_eval.post_train.latencies=[3]"
      )
      ;;
    demon_attack)
      latency=6
      episodes=200
      eval_args=(
        "rl_games.env_eval.mid_train.enabled=false"
        "rl_games.env_eval.post_train.enabled=false"
      )
      ;;
    deadly_corridor)
      latency=6
      episodes=1000
      eval_args=(
        "rl_games.env_eval.mid_train.enabled=false"
        "rl_games.env_eval.post_train.enabled=false"
      )
      ;;
  esac

  raw_name="${env_name}_fix_latency_${latency}_${episodes}ep_7k2steps"
  run_id="${raw_name}_${method}"
  dataset_local_dir="data/memory/${raw_name}"
  command="${STARVLA_ROOT}/commands/memory/train_flappy_openvla_${method}.sh"
  run_dir="${STARVLA_ROOT}/results/Checkpoints/${run_id}"

  conda run --no-capture-output -n starvla_rl_games_openvla bash \
    "${command}" \
    "env=${env_name}" \
    "run_id=${run_id}" \
    "paths.dataset_local_dir=${dataset_local_dir}" \
    "dataset.target_latency_unit=raw_frames" \
    "${eval_args[@]}"

  conda run --no-capture-output -n latency hf upload \
    "${VLA_REPO}" \
    "${run_dir}" \
    "./${run_id}" \
    --exclude "wandb/**" \
    --exclude "checkpoints/_initialization/**" \
    --exclude "checkpoints/*_state/**" \
    --exclude "**/optimizer.bin" \
    --exclude "**/*optim_states.pt" \
    --repo-type model
}

select_envs "${1:-all}"
select_methods "${2:-all}"

conda run --no-capture-output -n latency hf download \
  "${BASE_MODEL_REPO}" \
  --revision "${BASE_MODEL_REVISION}" \
  --local-dir "${BASE_MODEL_DIR}"
conda run --no-capture-output -n latency hf download \
  "${INITIALIZER_REPO}" \
  "${INITIALIZER_CHECKPOINT}" \
  --revision "${INITIALIZER_REVISION}" \
  --local-dir "${INITIALIZER_DIR}"

for env_name in "${ENVS[@]}"; do
  prepare_data "${env_name}"
  for method in "${METHODS[@]}"; do
    train_method "${env_name}" "${method}"
  done
done
