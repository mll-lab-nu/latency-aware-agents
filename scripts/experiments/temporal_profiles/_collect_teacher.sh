#!/usr/bin/env bash
# Identical collection stages for the three local profile-trained teachers.
set -euo pipefail
METHOD="${1:?Use collect_temporal.sh, collect_hmm.sh or collect_block.sh}"
shift
case "${METHOD}" in temporal|hmm|block) ;; *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;; esac
UPLOAD_ONLY=0
case "${1:-}" in
    -h|--help)
        cat <<'EOF'
Usage: TEACHER_BUNDLE=/absolute/path/to/METHOD/seed_42/bundle \
       bash scripts/experiments/temporal_profiles/collect_{temporal,hmm,block}.sh

Activate/create Conda latency -> selected LOCAL teacher + its bundled profile ->
H8 two-pass demonstrations -> LeRobot conversion/audit -> HF dataset upload.
No teacher weights or profiles are downloaded. Run collection on Linux/CUDA.

Environment variables:
  TEACHER_BUNDLE      Required local bundle from train_METHOD.sh (contains DONE,
                     training_complete.json, config.json, checkpoint_p0/, profile/)
  WORKSPACE_DIR       Parent workspace (default: parent of this checkout)
  OUTPUT_DIR          Shared dataset root; each method gets METHOD/seed_42
  GPU                 One physical CUDA device, default 0
  EPISODES            Initial episodes, default 200 (add 50 to reach row target)
  MAX_ENV_STEPS       Raw-frame cap per episode, default 7200
  MIN_TRAIN_ROWS      Minimum training labels, default 320000
  HF_DATASET_REPO     Default latency-sensitive-bench/temporal-profile-rollouts
  FORCE_INSTALL=1    Repair/create latency env using the existing installer
  SKIP_UPLOAD=1      Keep the completed dataset locally

Timing is simulated at 10/10 Hz with the teacher's own profile, capacity 1,
256 parallel environments, eight-decision history, 90/10 episode split, seed 42,
split seed 0, deterministic actions, no filtering, one current student RGB image.
7200 is the agreed comparison cap; current historical collector defaults to 3600.

Authenticate using HF_TOKEN or hf auth login. A new dataset repo is private;
existing visibility is preserved. No checkpoint weights are uploaded.
Rerunning a completed collection reuses its data. Partial GPU image replay
retains the existing collector's restart restriction; use a fresh OUTPUT_DIR.
After an upload failure, retry WITHOUT GPU collection or TEACHER_BUNDLE:
  OUTPUT_DIR=/same/root bash .../collect_METHOD.sh --upload-only
The shared output root must be the same as in the original invocation.
EOF
        exit 0 ;;
    --upload-only) UPLOAD_ONLY=1 ;;
    "") ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo "Unexpected extra arguments (see --help)" >&2; exit 2; fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
if [[ "${UPLOAD_ONLY}" != 1 ]]; then
    : "${TEACHER_BUNDLE:?Set TEACHER_BUNDLE to the local method/seed_42/bundle directory}"
    if [[ "$(uname -s)" != Linux ]]; then
        echo "Run Flappy demonstration collection on the Linux/CUDA machine." >&2
        exit 1
    fi
fi
RUN_DIR="${OUTPUT_DIR:?}/${METHOD}/seed_42"
mkdir -p "${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
export PYTHONUNBUFFERED=1

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
    if ! conda env list | awk '$1 == "latency" { found=1 } END { exit !found }'; then NEED_INSTALL=1; fi
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
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/sample-factory:${REPO_ROOT}/third_party/flappy-bird-gymnasium${PYTHONPATH:+:${PYTHONPATH}}"
MODULE=scripts.experiments.temporal_profiles.rollout

if [[ "${UPLOAD_ONLY}" != 1 ]]; then
    python -m "${MODULE}" prepare --output-dir "${RUN_DIR}" --method "${METHOD}" \
        --teacher-bundle "${TEACHER_BUNDLE}" --episodes "${EPISODES:-200}" \
        --max-steps "${MAX_ENV_STEPS:-7200}" --min-train-rows "${MIN_TRAIN_ROWS:-320000}" \
        2>&1 | tee -a "${RUN_DIR}/prepare.log"
    if [[ ! -f "${RUN_DIR}/DONE" ]]; then
        if [[ ! -f "${RUN_DIR}/collection_complete.json" ]]; then
            GPU="${GPU:-0}"
            if [[ -z "${GPU}" || "${GPU}" == *,* ]]; then echo "GPU must select one CUDA device." >&2; exit 2; fi
            export CUDA_VISIBLE_DEVICES="${GPU}"
            export CUDA_HOME="${CONDA_PREFIX}"
            export PATH="${CUDA_HOME}/bin:${PATH}"
            export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            export SDL_VIDEODRIVER=dummy
            export SDL_AUDIODRIVER=dummy
            export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${WORKSPACE_DIR}/outputs/torch_extensions/flappy_teachers}"
            python -m scripts.install.validate base
            python -m scripts.install.validate flappy
            bash scripts/bash_scripts/build_cuda_extensions.sh --env flappy
            python -m "${MODULE}" collect --output-dir "${RUN_DIR}" 2>&1 | tee -a "${RUN_DIR}/collect.log"
        fi
        python -m "${MODULE}" export --output-dir "${RUN_DIR}" 2>&1 | tee -a "${RUN_DIR}/export.log"
    fi
fi
if [[ "${UPLOAD_ONLY}" == 1 || "${SKIP_UPLOAD:-0}" != 1 ]]; then
    python -m "${MODULE}" upload --output-dir "${RUN_DIR}" \
        --repo-id "${HF_DATASET_REPO:-latency-sensitive-bench/temporal-profile-rollouts}" \
        2>&1 | tee -a "${RUN_DIR}/upload.log"
fi
echo "VLA training dataset: ${RUN_DIR}/lerobot"
echo "VLA validation dataset: ${RUN_DIR}/lerobot__val"
echo "Collection checks: ${RUN_DIR}/quality_report.json and history_audit.json"
