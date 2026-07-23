#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STARVLA_ROOT="${REPO_ROOT}/third_party/starVLA"
TEACHER_ROOT="${REPO_ROOT}/results/checkpoints/jitter_destructiveness/released/teachers"
RAW_ROOT="${REPO_ROOT}/teacher_datasets/jitter_destructiveness"

usage() {
  cat >&2 <<'EOF'
Usage: rollout_data.sh [flappy|demon_attack|deadly_corridor|all]
EOF
}

select_envs() {
  case "$1" in
    flappy|demon_attack|deadly_corridor) ENVS=("$1") ;;
    all) ENVS=(flappy demon_attack deadly_corridor) ;;
    *) usage; exit 2 ;;
  esac
}

rollout_env() {
  local env_name="$1"
  local teacher_name episodes raw_name output_root converter data_mix
  local converter_args=()
  case "${env_name}" in
    flappy)
      teacher_name="flappy_sf_same_latency_l2_seed0"
      episodes=200
      raw_name="flappy_fix_latency_2_200ep"
      output_root="${STARVLA_ROOT}/data/jitter_destructiveness/flappy_fix_latency_2_200ep"
      converter="convert_flappy_to_starvla_lerobot.py"
      data_mix="flappy_train__bridge"
      converter_args=(--action-carrier bridge)
      ;;
    demon_attack)
      teacher_name="demon_attack_sf_same_latency_l2_seed0"
      episodes=200
      raw_name="demon_attack_fix_latency_2_200ep"
      output_root="${STARVLA_ROOT}/data/jitter_destructiveness/demon_attack_fix_latency_2_200ep"
      converter="convert_demon_attack_to_starvla_lerobot.py"
      data_mix="demon_attack_train__bridge"
      converter_args=(--action-carrier bridge)
      ;;
    deadly_corridor)
      teacher_name="deadly_corridor_sf_same_latency_l2_seed0"
      episodes=1000
      raw_name="deadly_corridor_fix_latency_2_1000ep"
      output_root="${STARVLA_ROOT}/data/jitter_destructiveness/deadly_corridor_fix_latency_2"
      converter="convert_deadly_corridor_to_starvla_lerobot.py"
      data_mix="deadly_corridor_train__bridge"
      converter_args=(--action-carrier bridge --action-layout multibinary_7)
      ;;
  esac

  MAX_EPISODES="${episodes}" \
  MAX_STEPS=3600 \
  DEVICE=cuda \
  FILTER_PRESET=clean_v1 \
  conda run --no-capture-output -n latency bash \
    "${REPO_ROOT}/scripts/bash_scripts/run_sf_teacher_dataset_export.sh" \
    "${TEACHER_ROOT}/${teacher_name}" \
    "${RAW_ROOT}/${raw_name}"

  conda run --no-capture-output -n starvla_rl_games_openvla python \
    "${STARVLA_ROOT}/examples/rl_games/bash_scripts/gr00t/data_conversion/${converter}" \
    --dataset-name "${RAW_ROOT}/${raw_name}" \
    --output-dir "${output_root}/${data_mix}" \
    "${converter_args[@]}" \
    --force
}

select_envs "${1:-all}"
for env_name in "${ENVS[@]}"; do
  rollout_env "${env_name}"
done
