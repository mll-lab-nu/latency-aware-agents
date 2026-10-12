#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSITIONAL=()
SKIP_FLASH_ATTN="false"
SKIP_VALIDATE="false"

usage() {
  cat <<'EOF'
Usage: bash examples/rl_games/install/install_stack.sh [options] <model> [tasks]

Install fixed PyTorch 2.11/CUDA 12.8 model development and training environments.

Arguments:
  <model>                   openvla|pi0|pi05|gr00t|wan_oft|all
  [tasks]                   Comma-separated task names for training evaluation

Options:
  --skip-flash-attn        Skip FlashAttention install/check for model envs
  --skip-validate          Skip final validation
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-flash-attn)
      SKIP_FLASH_ATTN="true"
      shift
      ;;
    --skip-validate)
      SKIP_VALIDATE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

[[ ${#POSITIONAL[@]} -ge 1 && ${#POSITIONAL[@]} -le 2 ]]
MODEL_TARGET="${POSITIONAL[0]}"
case "${MODEL_TARGET}" in
  openvla|pi0|pi05|gr00t|wan_oft|all) ;;
  *) exit 1 ;;
esac

if [[ ${#POSITIONAL[@]} -eq 2 ]]; then
  ENV_TARGET="${POSITIONAL[1]}"
  if [[ "${ENV_TARGET}" == cross_task ]]; then
    ENV_TARGET="flappy,demon_attack,air_raid,deadly_corridor,asterix,atlantis"
  fi
fi

ARGS=(
  --tier dev
  --model "${MODEL_TARGET}"
)
if [[ ${#POSITIONAL[@]} -eq 2 ]]; then
  ARGS+=(--env "${ENV_TARGET}")
fi

if [[ "${SKIP_FLASH_ATTN}" == "true" ]]; then
  ARGS+=(--skip-flash-attn)
fi

if [[ "${SKIP_VALIDATE}" == "true" ]]; then
  ARGS+=(--skip-validate)
fi

exec "${INSTALL_DIR}/bootstrap.sh" "${ARGS[@]}"
