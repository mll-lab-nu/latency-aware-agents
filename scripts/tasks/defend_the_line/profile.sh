#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/tasks/defend_the_line/profile.sh <model> [MODEL_PATH] [--publish]
  bash scripts/tasks/defend_the_line/profile.sh <model> --hf-artifacts [--publish]
  bash scripts/tasks/defend_the_line/profile.sh <model> [MODEL_PATH] --resume RUN_ID

Models:
  openvla
  wanoft
  pi05
  gr00t

Runs the registered Defend the Line realtime profiling/evaluation workload:
5 sessions, 10 episodes per session, max 3600 env steps per episode.
Extra Hugging Face artifact overrides are forwarded to the shared profiler.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

MODEL="$1"
shift

case "${MODEL}" in
  openvla|wanoft|pi05|gr00t) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown Defend the Line profile model: ${MODEL}" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

args=(run --model "${MODEL}" --workload defend_the_line)
if [[ $# -gt 0 && "$1" != --* ]]; then
  args+=(--model-path "$1")
  shift
fi
args+=("$@")

exec bash scripts/profile.sh "${args[@]}"
