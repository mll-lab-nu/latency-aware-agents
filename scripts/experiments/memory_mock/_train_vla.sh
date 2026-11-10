#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
METHOD="${1:?Expected temporal, ar1 or arma}"
shift
case "${METHOD}" in temporal|ar1|arma) ;; *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;; esac
PREPARE_ONLY=0
EXTRA_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --prepare-only) PREPARE_ONLY=1; EXTRA_ARGS+=("${arg}") ;;
        --resume) EXTRA_ARGS+=("${arg}") ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/experiments/memory_mock/train_vla_{temporal,ar1,arma}.sh [--prepare-only | --resume]

Train QwenOFT using the main-paper hist8-v2 learning recipe: 5,000 updates,
1 GPU x batch 32 x accumulation 4, backend none, seed 42, H8, action horizon 1.
Mid-training environment evaluation is disabled. No live environment is run.
Train/eval data loaders use 2 workers each and prefetch 1 to reduce host RAM use.
Scheduled checkpoint saving and held-out validation at update 5,000 only.
Only the matching dataset and run identity differ between the three launchers.

Required:
  DATASET_ROOT     Parent containing {temporal,ar1,arma}/seed_42/lerobot.
                   Must be the completed, unfiltered 18,000-frame collections.
Optional:
  VLA_GPUS         One CUDA device ID (default: 0); independent of teacher GPU.
  VLA_OUTPUT_ROOT  Default: <workspace>/outputs/memory_mock/flappy_3090_vlas_20min_single_gpu
                   Independent of the collection's OUTPUT_DIR.
  WANDB_ENTITY     Your W&B account/team (otherwise W&B account default).
  WANDB_PROJECT    Default: flappy-irregular-vlas. WANDB_MODE=disabled disables it.
  VLA_PORT         Accelerate port (default: 29501).
  SKIP_INSTALL=1   Reuse installed starvla_rl_games_openvla training environment.

The normal run installs/activates the existing StarVLA dev environment and
downloads the exact paper Qwen backbone revision into the HF cache.
--prepare-only uses the current Python, validates data and saves train.yaml;
it performs no installation, download or GPU training. --resume restores the
latest optimizer/scheduler/RNG state. Checkpoints stay local; no HF upload.
EOF
            exit 0 ;;
        *) echo "Unknown argument: ${arg} (see --help)" >&2; exit 2 ;;
    esac
done

: "${DATASET_ROOT:?Set DATASET_ROOT to the parent of temporal/ar1/arma collection folders}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "${REPO_ROOT}")}"
VLA_OUTPUT_ROOT="${VLA_OUTPUT_ROOT:-${WORKSPACE_DIR}/outputs/memory_mock/flappy_3090_vlas_20min_single_gpu}"
export CUDA_VISIBLE_DEVICES="${VLA_GPUS:-0}"
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "Select one GPU for this launcher, for example: export VLA_GPUS=0" >&2
    exit 2
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/starVLA${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"

if [[ "${PREPARE_ONLY}" == 0 ]]; then
    if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
        bash install.sh --env flappy --model openvla --tier dev
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
    conda activate "${STARVLA_CONDA_ENV_NAME:-starvla_rl_games_openvla}"
    set -u
    export CUDA_HOME="${CONDA_PREFIX}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    BASE_MODEL_DIR="$(hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17)"
else
    BASE_MODEL_DIR="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    EXTRA_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

exec python -m scripts.experiments.memory_mock.train_vla \
    --method "${METHOD}" --collection-dir "${DATASET_ROOT}/${METHOD}/seed_42" \
    --base-model-dir "${BASE_MODEL_DIR}" --output-root "${VLA_OUTPUT_ROOT}" \
    --port "${VLA_PORT:-29501}" --wandb-project "${WANDB_PROJECT:-flappy-irregular-vlas}" \
    "${EXTRA_ARGS[@]}"
