#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${FINAL_PROFILE_LOG_DIR:-${SCRIPT_DIR}/logs}"
GPU_IDS_STR="${GPU_IDS:-0 1 2 3 4 5 6 7 8}"

read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

SCRIPTS=(
  flappy_openvla.sh
  flappy_pi05.sh
  flappy_gr00t.sh
  demon_attack_openvla.sh
  demon_attack_pi05.sh
  demon_attack_gr00t.sh
  deadly_corridor_openvla.sh
  deadly_corridor_pi05.sh
  deadly_corridor_gr00t.sh
)

if (( ${#GPU_IDS[@]} != ${#SCRIPTS[@]} )); then
  echo "[error] expected ${#SCRIPTS[@]} GPU ids, got ${#GPU_IDS[@]}: ${GPU_IDS_STR}" >&2
  echo "        Override with: GPU_IDS='0 1 2 3 4 5 6 7 8' bash $0" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

for idx in "${!SCRIPTS[@]}"; do
  script="${SCRIPTS[$idx]}"
  gpu="${GPU_IDS[$idx]}"
  name="${script%.sh}"
  log_path="${LOG_DIR}/${name}.log"
  echo "[launch] gpu=${gpu} ${script} -> ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" bash "${SCRIPT_DIR}/${script}" >"${log_path}" 2>&1 &
done

wait
