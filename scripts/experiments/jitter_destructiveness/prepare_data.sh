#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STARVLA_ROOT="${REPO_ROOT}/third_party/starVLA"
RAW_ROOT="${STARVLA_ROOT}/data/jitter_destructiveness/raw"

usage() {
  cat >&2 <<'EOF'
Usage: prepare_data.sh [flappy|demon_attack|deadly_corridor|all]
EOF
}

select_envs() {
  case "$1" in
    flappy|demon_attack|deadly_corridor) ENVS=("$1") ;;
    all) ENVS=(flappy demon_attack deadly_corridor) ;;
    *) usage; exit 2 ;;
  esac
}

prepare_flappy() {
  conda run --no-capture-output -n starvla_rl_games_openvla hf download \
    latency-sensitive-bench/flappy_200ep \
    --repo-type dataset \
    --revision 5df5688a961dc2c91090155c719bff4a5b248235 \
    --include 'flappy_fix_latency_2_200ep/**' \
    --local-dir "${RAW_ROOT}"
  conda run --no-capture-output -n starvla_rl_games_openvla python \
    "${STARVLA_ROOT}/examples/rl_games/bash_scripts/gr00t/data_conversion/convert_flappy_to_starvla_lerobot.py" \
    --dataset-name "${RAW_ROOT}" \
    --dataset-source-subdir flappy_fix_latency_2_200ep \
    --output-dir "${STARVLA_ROOT}/data/jitter_destructiveness/flappy_fix_latency_2_200ep/flappy_train__bridge" \
    --action-carrier bridge \
    --force
}

prepare_demon_attack() {
  conda run --no-capture-output -n starvla_rl_games_openvla hf download \
    latency-sensitive-bench/demon_attack_200ep \
    --repo-type dataset \
    --revision 0481972d94f6409a392c5b035f9a7f5c2681a96d \
    --include 'demon_attack_fix_latency_2_200ep/**' \
    --local-dir "${RAW_ROOT}"
  conda run --no-capture-output -n starvla_rl_games_openvla python \
    "${STARVLA_ROOT}/examples/rl_games/bash_scripts/gr00t/data_conversion/convert_demon_attack_to_starvla_lerobot.py" \
    --dataset-name "${RAW_ROOT}" \
    --dataset-source-subdir demon_attack_fix_latency_2_200ep \
    --output-dir "${STARVLA_ROOT}/data/jitter_destructiveness/demon_attack_fix_latency_2_200ep/demon_attack_train__bridge" \
    --action-carrier bridge \
    --force
}

prepare_deadly_corridor() {
  conda run --no-capture-output -n starvla_rl_games_openvla hf download \
    latency-sensitive-bench/deadly_1000ep_11 \
    --repo-type dataset \
    --revision 1c9240945cee42555cd1d725338b712cae42ec30 \
    --include 'deadly_corridor_fix_latency_2_1000ep/**' \
    --local-dir "${RAW_ROOT}"
  conda run --no-capture-output -n starvla_rl_games_openvla python \
    "${STARVLA_ROOT}/examples/rl_games/bash_scripts/gr00t/data_conversion/convert_deadly_corridor_to_starvla_lerobot.py" \
    --dataset-name "${RAW_ROOT}" \
    --dataset-source-subdir deadly_corridor_fix_latency_2_1000ep \
    --output-dir "${STARVLA_ROOT}/data/jitter_destructiveness/deadly_corridor_fix_latency_2/deadly_corridor_train__bridge" \
    --action-carrier bridge \
    --action-layout multibinary_7 \
    --force
}

select_envs "${1:-all}"
for env_name in "${ENVS[@]}"; do
  "prepare_${env_name}"
done
