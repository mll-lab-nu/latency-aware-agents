#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage: eval.sh <small|vla> [flappy|demon_attack|deadly_corridor|all] [released|trained]
EOF
}

select_envs() {
  case "$1" in
    flappy|demon_attack|deadly_corridor) ENVS=("$1") ;;
    all) ENVS=(flappy demon_attack deadly_corridor) ;;
    *) usage; exit 2 ;;
  esac
}

generate_small_configs() {
  local env_name="$1"
  conda run --no-capture-output -n latency python scripts/sweeps/generate_sweep_configs.py \
    --spec "configs/experiments/jitter_destructiveness/small_${env_name}.yaml"
}

eval_small() {
  local env_name="$1"
  local source="$2"
  local config_path seed output_dir
  generate_small_configs "${env_name}"
  while IFS= read -r config_path; do
    seed="$(basename "$(dirname "${config_path}")")"
    seed="${seed#seed_}"
    output_dir="outputs/jitter_destructiveness/${source}/small/${env_name}/seed_${seed}"
    if [[ "${source}" == "released" ]]; then
      conda run --no-capture-output -n latency python -m latency_bench.run \
        --config "${config_path}" \
        --checkpoint-path "results/checkpoints/jitter_destructiveness/released/small/${env_name}/seed_${seed}/checkpoint.pth" \
        --output-dir "${output_dir}"
    else
      conda run --no-capture-output -n latency python -m latency_bench.run \
        --config "${config_path}" \
        --output-dir "${output_dir}"
    fi
  done < <(find "configs/train/jitter_destructiveness/${env_name}" -type f -name latency_bench_eval.yaml | sort)
}

vla_checkpoint() {
  local env_name="$1"
  local source="$2"
  local checkpoint_root="results/checkpoints/jitter_destructiveness/${source}/vla"
  case "${env_name}" in
    flappy) echo "${checkpoint_root}/flappy_fix_latency_2_200ep/checkpoints/steps_5000_state/model.safetensors" ;;
    demon_attack) echo "${checkpoint_root}/demon_attack_fix_latency_2_200ep/checkpoints/steps_7000_state/model.safetensors" ;;
    deadly_corridor) echo "${checkpoint_root}/deadly_corridor_fix_latency_2/checkpoints/best_state/model.safetensors" ;;
  esac
}

eval_vla() {
  local env_name="$1"
  local source="$2"
  local config_path="configs/experiments/jitter_destructiveness/vla_${env_name}.yaml"
  local output_dir="outputs/jitter_destructiveness/${source}/vla/${env_name}"
  local checkpoint_path
  checkpoint_path="$(vla_checkpoint "${env_name}" "${source}")"
  conda run --no-capture-output -n starvla_rl_games_openvla python -m latency_bench.run \
    --config "${config_path}" \
    --checkpoint-path "${checkpoint_path}" \
    --output-dir "${output_dir}"
}

family="${1:-}"
env_selector="${2:-all}"
source="${3:-released}"
select_envs "${env_selector}"

case "${source}" in
  released|trained) ;;
  *) usage; exit 2 ;;
esac

case "${family}" in
  small)
    for env_name in "${ENVS[@]}"; do eval_small "${env_name}" "${source}"; done
    ;;
  vla)
    for env_name in "${ENVS[@]}"; do eval_vla "${env_name}" "${source}"; done
    ;;
  *)
    usage
    exit 2
    ;;
esac
