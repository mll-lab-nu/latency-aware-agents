#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GR00T_PROJECT="${REPO_ROOT}/third_party/Isaac-GR00T"
DATASET_DIR="${REPO_ROOT}/outputs/official_gr00t/datasets/humanoidbench"
BASE_MODEL_DIR="${REPO_ROOT}/outputs/official_gr00t/models/GR00T-N1.7-3B"
CHECKPOINT_DIR="${REPO_ROOT}/outputs/official_gr00t/g1_balance_simple_obs198_h40_repeat_fixedz0_onestep_dropout0_frompretrained_bs128_lr1e4_15k_periodic500/checkpoint-15000"
HF_REPO="latency-sensitive-bench/humanoidbench"

cd "${REPO_ROOT}"

uv run --project "${GR00T_PROJECT}" hf download "${HF_REPO}" \
    --repo-type dataset \
    --revision "${HF_DATASET_REVISION}" \
    --include "balance_simple/lerobot/**" "balance_simple/lerobot__val/**" \
    --local-dir "${DATASET_DIR}"
uv run --project "${GR00T_PROJECT}" hf download nvidia/GR00T-N1.7-3B \
    --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
    --local-dir "${BASE_MODEL_DIR}"

uv run --project "${GR00T_PROJECT}" python \
    scripts/official_gr00t/train_balance_simple_periodic_eval.py "$@"

uv run --project "${GR00T_PROJECT}" hf repo create "${HF_REPO}" \
    --repo-type model --exist-ok
uv run --project "${GR00T_PROJECT}" hf upload "${HF_REPO}" \
    "${CHECKPOINT_DIR}" balance_simple/checkpoint-15000 \
    --repo-type model \
    --include config.json embodiment_id.json processor_config.json statistics.json \
    "*.safetensors" "*.safetensors.index.json"
uv run --project "${GR00T_PROJECT}" hf upload "${HF_REPO}" \
    docs/humanoidbench/hf_model_card.md README.md \
    --repo-type model
