#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat >&2 <<'EOF'
Usage: train_small_policy.sh [flappy|demon_attack|deadly_corridor|all]
EOF
}

select_envs() {
  case "$1" in
    flappy|demon_attack|deadly_corridor) ENVS=("$1") ;;
    all) ENVS=(flappy demon_attack deadly_corridor) ;;
    *) usage; exit 2 ;;
  esac
}

generate_configs() {
  local env_name="$1"
  conda run --no-capture-output -n latency python scripts/sweeps/generate_sweep_configs.py \
    --spec "configs/experiments/jitter_destructiveness/small_${env_name}.yaml"
}

select_envs "${1:-all}"
for env_name in "${ENVS[@]}"; do
  generate_configs "${env_name}"
  while IFS= read -r config_path; do
    conda run --no-capture-output -n latency python -m training.train --config "${config_path}"
  done < <(find "configs/train/jitter_destructiveness/${env_name}" -type f -name train.yaml | sort)
done
