#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
METHOD="${1:?Expected temporal, ar1 or arma}"
shift
case "${METHOD}" in temporal|ar1|arma) ;; *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;; esac
EXTRA_ARGS=()
PREPARE_ONLY=0
CONTENTION=0
for arg in "$@"; do
    case "${arg}" in
        --prepare-only) PREPARE_ONLY=1; EXTRA_ARGS+=("${arg}") ;;
        --contention) CONTENTION=1; EXTRA_ARGS+=("${arg}") ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/experiments/memory_mock/eval_vla_{temporal,ar1,arma}.sh [--prepare-only] [--contention]

Download the selected step-5000 VLA and pinned Qwen backbone, then invoke the
existing latency_bench realtime evaluator. Flappy runs at 10/10 FPS with H8,
one image, one inference worker, and a latency-neutral prompt. No simulated
latency is used. Run models sequentially.
The *_contention.sh launchers add the original irregular GPU copy workload:
two 512 MiB buffers; OFF 5-15s; ON 80% 0.2-0.8s / 20% 2-5s. The existing
coordinator waits for model warmup and stops the workload with evaluation.

Defaults and overrides:
  GPU=0                    One physical GPU, exposed as logical cuda:0.
  EVAL_EPISODES=100         Episodes, ending on death or the frame cap.
  EVAL_MAX_STEPS=3600       Raw frames per episode (six minutes at 10 FPS).
  EVAL_SEED=1000000         Consecutive episode seeds start here.
  CONTENTION_SEED=20261110  Same fresh workload seed for all three methods.
  HF_MODEL_REPO=talha15032/flappy-irregular-qwenoft
  HF_MODEL_REVISION=main    Resolves to a fixed commit before evaluation.
  HF_HOME                  Optional HF cache location on a disk with space.
  EVAL_OUTPUT_ROOT         Default: <workspace>/outputs/memory_mock/realtime_vlas
                           Contention default ends in realtime_vlas_contention.
  EVAL_RUN_ID              Default: UTC timestamp; use a fresh ID for each run.
  STARVLA_CONDA_ENV_NAME    Default: starvla_rl_games_openvla.
  SKIP_INSTALL=1           Reuse the profiling environment (default).
                           Set 0 to run the existing installer first.

The selected HF folder must contain checkpoints/steps_5000_pytorch_model.pt,
config.full.yaml and dataset_statistics.json. Upload AR1 before evaluating it.
The dataset and optimizer state are not downloaded. No results are uploaded.
--prepare-only uses the current Python to download and save the evaluation
configuration, without activating/installing an environment or running a GPU.
Each output directory is new; this launcher does not resume partial evaluations.
EOF
            exit 0 ;;
        *) echo "Unknown argument: ${arg} (see --help)" >&2; exit 2 ;;
    esac
done
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
if [[ -z "${CUDA_VISIBLE_DEVICES}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "GPU must select exactly one physical device." >&2
    exit 2
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/starVLA${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
if [[ "${PREPARE_ONLY}" == 0 ]]; then
    if [[ "${SKIP_INSTALL:-1}" != 1 ]]; then
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
    conda activate "${STARVLA_CONDA_ENV_NAME:-starvla_rl_games_openvla}"
    set -u
    export CUDA_HOME="${CONDA_PREFIX}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
WORKSPACE_DIR="${WORKSPACE_DIR:-$(dirname "${REPO_ROOT}")}"
DEFAULT_OUTPUT_ROOT="${WORKSPACE_DIR}/outputs/memory_mock/realtime_vlas"
if [[ "${CONTENTION}" == 1 ]]; then
    DEFAULT_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}_contention"
    EXTRA_ARGS+=(--load-seed "${CONTENTION_SEED:-20261110}")
fi
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
EVAL_RUN_ID="${EVAL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
exec python -m scripts.experiments.memory_mock.eval_vla \
    --method "${METHOD}" --repo-id "${HF_MODEL_REPO:-talha15032/flappy-irregular-qwenoft}" \
    --revision "${HF_MODEL_REVISION:-main}" \
    --episodes "${EVAL_EPISODES:-100}" --max-steps "${EVAL_MAX_STEPS:-3600}" \
    --seed "${EVAL_SEED:-1000000}" \
    --output-dir "${EVAL_OUTPUT_ROOT}/${METHOD}/${EVAL_RUN_ID}" "${EXTRA_ARGS[@]}"
