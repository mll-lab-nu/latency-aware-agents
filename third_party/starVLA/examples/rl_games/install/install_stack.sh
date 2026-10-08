#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POSITIONAL=()
SKIP_FLASH_ATTN="false"
SKIP_VALIDATE="false"

usage() {
  cat <<'EOF'
Usage: bash examples/rl_games/install/install_stack.sh [options] <model> [legacy-env]

Compatibility entrypoint for the fixed PyTorch 2.11/CUDA 12.8 model
development and training environments. Every model environment installs the
shared RL-games stack.

Arguments:
  <model>                   openvla|pi0|pi05|gr00t|wan_oft|all
  [legacy-env]              Accepted for existing launch scripts; installation
                            still includes the shared RL-games stack

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
  case "${POSITIONAL[1]}" in
    flappy|demon_attack|defend_the_line|deadly_corridor|asterix|atlantis|cross_task) ;;
    *) exit 1 ;;
  esac
  echo "[install_stack] legacy env selector '${POSITIONAL[1]}' accepted; installing all games"
fi

ARGS=(
  --tier dev
  --model "${MODEL_TARGET}"
)

if [[ "${SKIP_FLASH_ATTN}" == "true" ]]; then
  ARGS+=(--skip-flash-attn)
fi

if [[ "${SKIP_VALIDATE}" == "true" ]]; then
  ARGS+=(--skip-validate)
fi

exec "${INSTALL_DIR}/bootstrap.sh" "${ARGS[@]}"
