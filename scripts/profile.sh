#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

MODEL=""
PREVIOUS=""
for ARG in "$@"; do
  if [[ "${PREVIOUS}" == "--model" ]]; then
    MODEL="${ARG}"
    break
  fi
  case "${ARG}" in
    --model=*) MODEL="${ARG#--model=}"; break ;;
  esac
  PREVIOUS="${ARG}"
done
CONDA_ENV="${MODEL:+starvla_rl_games_${MODEL}}"
CONDA_ENV="${CONDA_ENV:-latency}"

set +u
if [[ -d "/venv/${CONDA_ENV}" ]]; then
  conda activate "/venv/${CONDA_ENV}"
else
  conda activate "${CONDA_ENV}"
fi
set -u

python -m latency_bench.cross_device.protocol "$@"
