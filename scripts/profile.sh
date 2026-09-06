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
if [[ "${MODEL}" == "official_gr00t" || -z "${MODEL}" ]]; then
  CONDA_ENV="latency"
elif [[ "${MODEL}" == "qwenoft" ]]; then
  CONDA_ENV="starvla_rl_games_openvla"
else
  CONDA_ENV="starvla_rl_games_${MODEL}"
fi
export LATENCY_BENCH_ENV_BACKEND_START_METHOD=spawn

set +u
if [[ -d "/venv/${CONDA_ENV}" ]]; then
  conda activate "/venv/${CONDA_ENV}"
else
  CONDA_ENV_PREFIX="$(conda env list | awk -v suffix="/${CONDA_ENV}" '$NF ~ suffix "$" { print $NF; exit }')"
  if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
    conda activate "${CONDA_ENV_PREFIX}"
  else
    conda activate "${CONDA_ENV}"
  fi
fi
set -u

python -m latency_bench.cross_device.protocol "$@"
