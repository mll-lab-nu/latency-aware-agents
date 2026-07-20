#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STARVLA_ROOT="${REPO_ROOT}/third_party/starVLA"
RUN_ROOT="${REPO_ROOT}/results/checkpoints/jitter_destructiveness/trained/vla"
BASE_MODEL_REPO="Qwen/Qwen3-VL-4B-Instruct"
BASE_MODEL_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
BASE_MODEL_DIR="${STARVLA_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
INITIALIZER_REPO="StarVLA/Qwen3VL-OFT-Bridge-RT-1"
INITIALIZER_REVISION="c3fc8f028429ba14819bf3b16e098776b670c889"
INITIALIZER_CHECKPOINT="checkpoints/steps_5000_pytorch_model.pt"
INITIALIZER_DIR="${STARVLA_ROOT}/playground/Pretrained_models/Qwen3VL-OFT-Bridge-RT-1"

usage() {
  cat >&2 <<'EOF'
Usage: train_vla.sh [flappy|demon_attack|deadly_corridor|all]
EOF
}

select_envs() {
  case "$1" in
    flappy|demon_attack|deadly_corridor) ENVS=("$1") ;;
    all) ENVS=(flappy demon_attack deadly_corridor) ;;
    *) usage; exit 2 ;;
  esac
}

select_envs "${1:-all}"
conda run --no-capture-output -n latency hf download \
  "${BASE_MODEL_REPO}" \
  --revision "${BASE_MODEL_REVISION}" \
  --local-dir "${BASE_MODEL_DIR}"
conda run --no-capture-output -n latency hf download \
  "${INITIALIZER_REPO}" \
  "${INITIALIZER_CHECKPOINT}" \
  --revision "${INITIALIZER_REVISION}" \
  --local-dir "${INITIALIZER_DIR}"

cd "${STARVLA_ROOT}"
for env_name in "${ENVS[@]}"; do
  conda run --no-capture-output -n starvla_rl_games_openvla python \
    examples/rl_games/scripts/launch_train.py \
    --config-name "jitter_destructiveness/${env_name}" \
    --run-root-dir "${RUN_ROOT}"
done
