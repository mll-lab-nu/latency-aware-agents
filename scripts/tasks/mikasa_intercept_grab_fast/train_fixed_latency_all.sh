#!/usr/bin/env bash
# Run all InterceptGrabFast fixed-latency small PPO teacher jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-train}"
shift || true

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/tasks/mikasa_intercept_grab_fast/train_fixed_latency_all.sh [train|eval|both] [GPU:LATENCIES ...]

Examples:
  # Default: GPU 0 runs L0,L2,L4; GPU 1 runs L1,L3.
  bash scripts/tasks/mikasa_intercept_grab_fast/train_fixed_latency_all.sh train

  # Same thing, explicitly:
  bash scripts/tasks/mikasa_intercept_grab_fast/train_fixed_latency_all.sh train 0:0,2,4 1:1,3

  # Run all five sequentially on one GPU:
  bash scripts/tasks/mikasa_intercept_grab_fast/train_fixed_latency_all.sh train 0:0,1,2,3,4

Each GPU:LATENCIES group is one sequential queue. Different groups run in parallel.
USAGE
}

case "${MODE}" in
  train|eval|both) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "$#" -eq 0 ]]; then
  set -- "0:0,2,4" "1:1,3"
fi

run_queue() {
  local spec="$1"
  local gpu="${spec%%:*}"
  local latencies="${spec#*:}"
  local latency

  if [[ -z "${gpu}" || "${gpu}" == "${spec}" || -z "${latencies}" ]]; then
    echo "Bad queue spec '${spec}'. Expected GPU:LATENCIES, for example 0:0,2,4." >&2
    exit 2
  fi

  IFS=',' read -r -a latency_list <<< "${latencies}"
  for latency in "${latency_list[@]}"; do
    latency="${latency#L}"
    latency="${latency#l}"
    case "${latency}" in
      0|1|2|3|4) ;;
      *)
        echo "Unsupported latency '${latency}' in queue '${spec}'." >&2
        exit 2
        ;;
    esac

    echo "[all] gpu=${gpu} latency=L${latency} mode=${MODE}"
    bash "${SCRIPT_DIR}/train_fixed_latency_l${latency}.sh" "${gpu}" "${MODE}"
  done
}

pids=()
for spec in "$@"; do
  run_queue "${spec}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
