#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "${REPO_ROOT}")}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/outputs/temporal_profiles/profiling/$(date -u +%Y%m%dT%H%M%SZ)}"
cd "${REPO_ROOT}"

case "${1:-}" in
    -h|--help)
        cat <<'EOF'
Usage: bash scripts/experiments/temporal_profiles/profile.sh

Install -> activate -> download zero-latency QwenOFT -> measure once ->
fit Temporal/HMM/block on common sessions -> export -> upload to HF.
Fresh measurement requires Linux/CUDA. Profiling uses 30/30 Hz, one worker.

Environment variables:
  WORKSPACE_DIR          Default: parent of this checkout
  OUTPUT_DIR             Default: $WORKSPACE_DIR/outputs/temporal_profiles/profiling/UTC_TIMESTAMP
  GPU                    One physical CUDA device (default: 0)
  REPEATS                Independent measurement processes (default: 5)
  EPISODES               Measured episodes per process (default: 10)
  MAX_ENV_STEPS          Raw-frame cap per profiling episode (default: 3600)
  FIT_SESSION_IDS        Comma-separated fitting IDs (default: 0,1,2)
  METHODS                Comma-separated methods (default: temporal,hmm,block)
  HMM_STATES             Default: 2; explicit setting, not automatically tuned
  HMM_MAX_ITERATIONS     Default: 1000 (three fitting restarts)
  BLOCK_MEAN_LENGTH      Default: 32 admitted requests; not automatically tuned
  EXISTING_RUN_DIR       Reuse completed raw sessions; skip download/measurement
  HF_REPO                Existing dataset repo (default: MLL-Lab/LAGEN-profiles)
  HF_PATH_PREFIX         Default: experiments/temporal-alternatives
  SKIP_INSTALL=1         Reuse the installed Conda environment and dependencies
  SKIP_UPLOAD=1          Export locally without uploading

Sessions outside FIT_SESSION_IDS are reserved and never used by these fits.
This script does not tune hyperparameters or evaluate timing fidelity/game scores.
Authenticate with HF_TOKEN or `hf auth login` before running with upload enabled.
Rerun with the SAME OUTPUT_DIR and settings to resume. To change a split or fit
setting, use a NEW OUTPUT_DIR and EXISTING_RUN_DIR from measurement.json.
The later demonstration-collection cap of 7200 is separate from this budget.
EOF
        exit 0 ;;
    "") ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
esac

GPU="${GPU:-0}"
if [[ -z "${GPU}" || "${GPU}" == *,* ]]; then
    echo "GPU must select exactly one device." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
MODEL_ENV="${STARVLA_CONDA_ENV_NAME:-starvla_rl_games_openvla}"

# 1. Reuse the maintained installer, then activate its model environment.
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
    bash install.sh --env flappy --model openvla --tier use
fi
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$(dirname "$(dirname "${CONDA_EXE}")")"
else
    CONDA_BASE="${HOME}/miniconda3"
fi
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${MODEL_ENV}"
set -u
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/starVLA${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
    python -m pip install -e '.[profiling,analysis]'
fi

IFS=',' read -r -a METHODS_ARRAY <<< "${METHODS:-temporal,hmm,block}"
IFS=',' read -r -a FIT_IDS_ARRAY <<< "${FIT_SESSION_IDS:-0,1,2}"
SOURCE_ARGS=()
if [[ -n "${EXISTING_RUN_DIR:-}" ]]; then
    SOURCE_ARGS+=(--existing-run-dir "${EXISTING_RUN_DIR}")
fi
mkdir -p "${OUTPUT_DIR}"
MODULE=scripts.experiments.temporal_profiles.profiling

# 2. Freeze the protocol before measurement and download only required assets.
python -m "${MODULE}" prepare --output-dir "${OUTPUT_DIR}" \
    --repeats "${REPEATS:-5}" --episodes "${EPISODES:-10}" \
    --max-env-steps "${MAX_ENV_STEPS:-3600}" \
    --methods "${METHODS_ARRAY[@]}" --fit-session-ids "${FIT_IDS_ARRAY[@]}" \
    --hmm-states "${HMM_STATES:-2}" --hmm-max-iterations "${HMM_MAX_ITERATIONS:-1000}" \
    --block-mean-length "${BLOCK_MEAN_LENGTH:-32}" "${SOURCE_ARGS[@]}" \
    2>&1 | tee -a "${OUTPUT_DIR}/prepare.log"

# 3. Record once; an interrupted run resumes its completed sessions.
python -m "${MODULE}" measure --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee -a "${OUTPUT_DIR}/measurement.log"

# 4. Fit each selected generator on the SAME sessions and export the bundle.
python -m "${MODULE}" export --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee -a "${OUTPUT_DIR}/fit.log"

# 5. Publish a new experiment folder; preserve the HF commit in a local receipt.
if [[ "${SKIP_UPLOAD:-0}" != 1 ]]; then
    python -m "${MODULE}" upload --output-dir "${OUTPUT_DIR}" \
        --repo-id "${HF_REPO:-MLL-Lab/LAGEN-profiles}" \
        --path-prefix "${HF_PATH_PREFIX:-experiments/temporal-alternatives}" \
        2>&1 | tee -a "${OUTPUT_DIR}/upload.log"
fi
echo "Profiles: ${OUTPUT_DIR}/profiles/<method>/profile.json"
echo "Exported bundle: ${OUTPUT_DIR}/bundle"
