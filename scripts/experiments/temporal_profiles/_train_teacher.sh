#!/usr/bin/env bash
# Common implementation keeps the three teacher recipes identical.
set -euo pipefail
METHOD="${1:?Use train_temporal.sh, train_hmm.sh or train_block.sh}"
shift
case "${METHOD}" in temporal|hmm|block) ;; *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;; esac
case "${1:-}" in
    -h|--help)
        cat <<'EOF'
Usage: bash scripts/experiments/temporal_profiles/train_{temporal,hmm,block}.sh

Check/create Conda latency env -> activate -> download this method's exact
uploaded Flappy/3090 profile -> train 25M APPO steps -> save -> upload model.
Requires Linux, a CUDA GPU and HF read/write access. No VLA weights are needed.

Environment variables:
  WORKSPACE_DIR       Parent workspace (default: parent of this checkout)
  OUTPUT_DIR          Shared root for all methods (each gets METHOD/seed_SEED)
  GPU                 One physical CUDA device, default 0
  SEED                Matched training/profile seed, default 42
  WANDB_ENTITY        Required W&B username or team
  WANDB_PROJECT       Optional override of YAML project flappy-teachers
  HF_MODEL_REPO       Default latency-sensitive-bench/temporal-profile-teachers
  FORCE_INSTALL=1     Repair/update the existing latency env with install.sh
  SKIP_UPLOAD=1       Keep completed checkpoints locally

Authenticate using HF_TOKEN or hf auth login. The model repo is created private
if absent; existing visibility is preserved. Run one script at a time per GPU.
W&B is enabled in the YAML. Authenticate with WANDB_API_KEY or wandb login.
Profiles are pinned to the uploaded flappy_3090_run1 revision: Temporal v13,
HMM 2 states (original fit), block mean 32. No offline-selected refit is substituted.
Rerun with the same OUTPUT_DIR/SEED to resume, including retrying failed uploads.
No code is committed or pushed. Demonstration collection is a later stage.
EOF
        exit 0 ;;
    "") ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
if [[ "$(uname -s)" != Linux ]]; then
    echo "Teacher training requires Linux/CUDA. Run this launcher on the GPU machine." >&2
    exit 1
fi
GPU="${GPU:-0}"
SEED="${SEED:-42}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY to your W&B username or team}"
if [[ -z "${GPU}" || "${GPU}" == *,* ]]; then
    echo "GPU must select exactly one CUDA device." >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
RUN_DIR="${OUTPUT_DIR:?}/${METHOD}/seed_${SEED}"
mkdir -p "${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

# 1. Locate Conda. The maintained installer bootstraps it if absent.
find_conda_sh() {
    local candidate
    local -a candidates=()
    if [[ -n "${CONDA_EXE:-}" ]]; then
        candidates+=("$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh")
    fi
    if command -v conda >/dev/null 2>&1; then
        candidates+=("$(conda info --base)/etc/profile.d/conda.sh")
    fi
    candidates+=("${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh")
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then printf '%s\n' "${candidate}"; return 0; fi
    done
    return 1
}

NEED_INSTALL="${FORCE_INSTALL:-0}"
if CONDA_SH="$(find_conda_sh)"; then
    set +u
    source "${CONDA_SH}"
    set -u
    if ! conda env list | awk '$1 == "latency" { found=1 } END { exit !found }'; then
        NEED_INSTALL=1
    fi
else
    NEED_INSTALL=1
fi
if [[ "${NEED_INSTALL}" == 1 ]]; then
    bash install.sh --env flappy --tier use 2>&1 | tee -a "${RUN_DIR}/install.log"
    CONDA_SH="$(find_conda_sh)"
fi
set +u
source "${CONDA_SH}"
conda activate latency
set -u
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/sample-factory:${REPO_ROOT}/third_party/flappy-bird-gymnasium${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${WORKSPACE_DIR}/outputs/torch_extensions/flappy_teachers}"
python -m scripts.install.validate base
python -m scripts.install.validate flappy
bash scripts/bash_scripts/build_cuda_extensions.sh --env flappy

MODULE=scripts.experiments.temporal_profiles.teacher
# 2. Download only this method's bundle and generate the existing training recipe.
PREPARE_ARGS=(--method "${METHOD}" --seed "${SEED}" --output-dir "${RUN_DIR}"
              --wandb-entity "${WANDB_ENTITY}")
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    PREPARE_ARGS+=(--wandb-project "${WANDB_PROJECT}")
fi
python -m "${MODULE}" prepare "${PREPARE_ARGS[@]}" \
    2>&1 | tee -a "${RUN_DIR}/prepare.log"

# 3. Resume interrupted training; skip it when a completed export already exists.
if [[ ! -f "${RUN_DIR}/bundle/DONE" ]]; then
    python -m training.train --config "${RUN_DIR}/train.yaml" \
        2>&1 | tee -a "${RUN_DIR}/train.log"
fi

# 4. Check actual checkpoint step counts, preserve best/latest/milestone weights.
python -m "${MODULE}" export --output-dir "${RUN_DIR}" \
    2>&1 | tee -a "${RUN_DIR}/export.log"

# 5. Upload a new model folder. A failed upload can be retried without retraining.
if [[ "${SKIP_UPLOAD:-0}" != 1 ]]; then
    python -m "${MODULE}" upload --output-dir "${RUN_DIR}" \
        --repo-id "${HF_MODEL_REPO:-latency-sensitive-bench/temporal-profile-teachers}" \
        2>&1 | tee -a "${RUN_DIR}/upload.log"
fi
echo "Saved ${METHOD} teacher: ${RUN_DIR}/bundle"
echo "Training and profile provenance: ${RUN_DIR}/protocol.json"
