#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${VLA_COLLECT_PY:-/home/ubuntu/miniconda3/envs/latency/bin/python}"
OUTPUT_ROOT="${VLA_DATA_ROOT:-${REPO_ROOT}/outputs/tasks/decision_history_vla_hist8/data}"
LOG_ROOT="${VLA_LOG_ROOT:-${REPO_ROOT}/outputs/tasks/decision_history_vla_hist8/logs}"
mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"
export TORCH_EXTENSIONS_DIR="${REPO_ROOT}/.cache/torch_extensions"

jobs=(
  flappy:qwengr00t:0
  flappy:qwenpi_v3:1
  demon_attack:qwenoft:2
  demon_attack:qwengr00t:3
  demon_attack:qwenpi_v3:4
  deadly_corridor:qwenoft:
  deadly_corridor:qwengr00t:
  deadly_corridor:qwenpi_v3:
)

pids=()
for job in "${jobs[@]}"; do
  IFS=: read -r task model gpu <<< "${job}"
  log="${LOG_ROOT}/collect-${task}-${model}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m scripts.tasks.decision_history.collect_vla_hist8 \
    --task "${task}" --model "${model}" \
    --output-root "${OUTPUT_ROOT}" --profile-root "${REPO_ROOT}" > "${log}" 2>&1 &
  pids+=("$!")
  printf '%s %s %s\n' "${task}" "${model}" "${pids[-1]}"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
exit "${failed}"
