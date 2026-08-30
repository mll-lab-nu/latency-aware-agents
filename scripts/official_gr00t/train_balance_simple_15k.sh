#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GR00T_PROJECT="${REPO_ROOT}/third_party/Isaac-GR00T"
GR00T_BIN="${GR00T_PROJECT}/.venv/bin"
CONDA_ENV_NAME="latency"
DATASET_DIR="${REPO_ROOT}/outputs/official_gr00t/datasets/humanoidbench"
BASE_MODEL_DIR="${REPO_ROOT}/outputs/official_gr00t/models/GR00T-N1.7-3B"
OUTPUT_DIR="${REPO_ROOT}/outputs/official_gr00t/g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_15k"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-15000"
DATASET_REVISION="dbb954842942196e127c9acc0d776b62aa679990"
DATASET_REPO="latency-sensitive-bench/humanoidbench"
ASSETS_REPO="latency-sensitive-bench/g1-sonic-assets"
ASSETS_REVISION="7c0a6616c78106adf9fa54dfc40e56ea07dc7d5d"
ASSETS_DIR="${REPO_ROOT}/results/humanoidbench_g1_sonic_assets"
BASE_MODEL_REPO="nvidia/GR00T-N1.7-3B"
BASE_MODEL_REVISION="2fc962b973bccdd5d8ce4f67cc63b264d6886495"
MODEL_REPO="latency-sensitive-bench/humanoidbench"

cd "${REPO_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${GR00T_BIN}/hf" download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${DATASET_REVISION}" \
    --include "balance_simple/lerobot/**" "balance_simple/lerobot__val/**" \
    --local-dir "${DATASET_DIR}"
"${GR00T_BIN}/hf" download "${ASSETS_REPO}" \
    --repo-type dataset \
    --revision "${ASSETS_REVISION}" \
    --include "sonic_g1/**" \
    --local-dir "${ASSETS_DIR}/assets"
"${GR00T_BIN}/hf" download "${BASE_MODEL_REPO}" \
    --revision "${BASE_MODEL_REVISION}" \
    --local-dir "${BASE_MODEL_DIR}"

"${GR00T_BIN}/python" \
    scripts/official_gr00t/train_balance_simple_periodic_eval.py "$@"

"${GR00T_BIN}/hf" repo create "${MODEL_REPO}" \
    --repo-type model --exist-ok
"${GR00T_BIN}/hf" upload "${MODEL_REPO}" \
    "${CHECKPOINT_DIR}" balance_simple/checkpoint-15000 \
    --repo-type model \
    --include config.json embodiment_id.json processor_config.json statistics.json \
    "*.safetensors" "*.safetensors.index.json"
"${GR00T_BIN}/hf" upload "${MODEL_REPO}" \
    docs/humanoidbench/hf_model_card.md README.md \
    --repo-type model
