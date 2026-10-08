#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_deadly_corridor_wan_oft_pipeline.sh [options]

Runs the fixed-raw-frame-latency-6 Deadly Corridor WanOFT pipeline:
  1. install/update the starvla_rl_games_wan_oft env
  2. download WanOFT checkpoints
  3. convert canonical benchmark row history directly into StarVLA LeRobot data
  4. train WanOFT through commands/wanoft/train_deadly_corridor_wan_oft.sh
  5. upload the run directory

The source latency is 6 raw VizDoom frames, or 1.5 policy decision steps at
frameskip=4. Core environment evaluation is disabled because the current
integer decision-step queue cannot reproduce a 1.5-step latency.

Options:
  --context-window <N>        Context window size (default: 5)
  --max-episodes <N>          Maximum source episodes per split (default: 1000)
  --max-train-steps <N>       Training steps (default: 2000)
  --benchmark-root <path>     latency-sensitive-bench checkout (default: sibling checkout)
  --dataset-cache-dir <path>  Optional Hugging Face cache directory
  --upload-repo <repo>        HF model repo for run upload (default: latency-sensitive-bench/deadly_corridor_1000ep)
  --upload-path <path>        Path inside the HF repo (default: <run_id>)
  --run-id <id>               Override run id
  --skip-env-setup            Do not run examples/rl_games/install/bootstrap.sh
  --skip-checkpoints          Do not download Wan base/init checkpoints
  --skip-convert              Do not convert source data
  --skip-train                Do not run training
  --skip-upload               Do not upload the run directory
  -h, --help                  Show this help
EOF
}

CONDA_ENV_NAME="starvla_rl_games_wan_oft"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-5}"
MAX_EPISODES="${MAX_EPISODES:-1000}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))}"
WANDB_ENTITY_VALUE="${WANDB_ENTITY:-dongqianyu99-zhejiang-university}"
WANDB_PROJECT_VALUE="${WANDB_PROJECT:-starVLA_rl_games}"
DATASET_REPO="latency-sensitive-bench/benchmark-datasets"
DATASET_REVISION="8a5af732702f31824156975ed248ce6534e736ee"
DATASET_CONFIG="deadly_corridor_fixed_latency_6_1000ep_7k2steps"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-}"
BENCHMARK_ROOT="${LATENCY_BENCH_ROOT:-}"
LATENCY_RAW_FRAMES=6
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"
RUN_ID="${RUN_ID:-}"
UPLOAD_REPO="${UPLOAD_REPO:-latency-sensitive-bench/deadly_corridor_1000ep}"
UPLOAD_PATH_IN_REPO="${UPLOAD_PATH_IN_REPO:-}"
BASE_MODEL_REPO="${BASE_MODEL_REPO:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
INIT_CHECKPOINT_REPO="${INIT_CHECKPOINT_REPO:-StarVLA/WM4A-Wan2d2-OFT-LIBERO-4in1}"
INIT_CHECKPOINT_DIR="${INIT_CHECKPOINT_DIR:-playground/Pretrained_models/WM4A-Wan2d2-OFT-LIBERO-4in1}"
SKIP_ENV_SETUP="false"
SKIP_CHECKPOINTS="false"
SKIP_CONVERT="false"
SKIP_TRAIN="false"
SKIP_UPLOAD="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --context-window)
      CONTEXT_WINDOW="$2"
      shift 2
      ;;
    --max-episodes)
      MAX_EPISODES="$2"
      shift 2
      ;;
    --max-train-steps)
      MAX_TRAIN_STEPS="$2"
      shift 2
      ;;
    --benchmark-root)
      BENCHMARK_ROOT="$2"
      shift 2
      ;;
    --dataset-cache-dir)
      DATASET_CACHE_DIR="$2"
      shift 2
      ;;
    --upload-repo)
      UPLOAD_REPO="$2"
      shift 2
      ;;
    --upload-path)
      UPLOAD_PATH_IN_REPO="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --skip-env-setup)
      SKIP_ENV_SETUP="true"
      shift
      ;;
    --skip-checkpoints)
      SKIP_CHECKPOINTS="true"
      shift
      ;;
    --skip-convert)
      SKIP_CONVERT="true"
      shift
      ;;
    --skip-train)
      SKIP_TRAIN="true"
      shift
      ;;
    --skip-upload)
      SKIP_UPLOAD="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[deadly-wanoft-fixed] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${CONTEXT_WINDOW}" =~ ^[0-9]+$ ]] || [[ "${CONTEXT_WINDOW}" -lt 2 ]]; then
  echo "[deadly-wanoft-fixed] --context-window must be an integer >= 2, got: ${CONTEXT_WINDOW}" >&2
  exit 2
fi
if ! [[ "${MAX_EPISODES}" =~ ^[0-9]+$ ]] || [[ "${MAX_EPISODES}" -lt 1 ]]; then
  echo "[deadly-wanoft-fixed] --max-episodes must be a positive integer, got: ${MAX_EPISODES}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${BENCHMARK_ROOT}" ]]; then
  BENCHMARK_ROOT="$(git -C "${REPO_ROOT}" rev-parse --show-superproject-working-tree)"
  if [[ -z "${BENCHMARK_ROOT}" ]]; then
    BENCHMARK_ROOT="${REPO_ROOT}/../latency-sensitive-bench"
  fi
fi
CONVERTED_DATA_ROOT="data/deadly_corridor_fix_latency_${LATENCY_RAW_FRAMES}_${MAX_EPISODES}ep_context${CONTEXT_WINDOW}"
CONVERTED_DATA_DIR="${CONVERTED_DATA_ROOT}/deadly_corridor_train__bridge"
PROMPT_MAP_PATH="${CONVERTED_DATA_DIR}/latency_prompt_map.json"
MANIFEST_PATH="${CONVERTED_DATA_DIR}/manifest.json"
RUN_ID="${RUN_ID:-wan_oft_deadly_corridor_fix_latency_${LATENCY_RAW_FRAMES}_context${CONTEXT_WINDOW}_standard_sft_${MAX_TRAIN_STEPS}_effbs${EFFECTIVE_BATCH_SIZE}_224_currentbce}"
UPLOAD_PATH_IN_REPO="${UPLOAD_PATH_IN_REPO:-${RUN_ID}}"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"

ensure_hf_cli() {
  if ! command -v hf >/dev/null 2>&1; then
    echo "[deadly-wanoft-fixed] Hugging Face CLI command 'hf' is not available in PATH." >&2
    echo "[deadly-wanoft-fixed] Re-run without --skip-env-setup, or install huggingface-hub in ${CONDA_ENV_NAME}." >&2
    exit 1
  fi
}

activate_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[deadly-wanoft-fixed] conda is required but was not found in PATH." >&2
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
}

validate_converted_dataset() {
  python - \
    "${MANIFEST_PATH}" \
    "${PROMPT_MAP_PATH}" \
    "${DATASET_CONFIG}" \
    "${CONTEXT_WINDOW}" \
    "${MAX_EPISODES}" <<'PY'
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
prompt_map_path = Path(sys.argv[2])
expected_config = sys.argv[3]
expected_context_window = int(sys.argv[4])
expected_max_episodes = int(sys.argv[5])

if not manifest_path.is_file():
    raise FileNotFoundError(f"Converted manifest does not exist: {manifest_path}")
if not prompt_map_path.is_file():
    raise FileNotFoundError(f"Converted prompt map does not exist: {prompt_map_path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
prompt_map = json.loads(prompt_map_path.read_text(encoding="utf-8"))
if manifest.get("source_config") != expected_config:
    raise ValueError(
        f"Expected source_config={expected_config!r}, got {manifest.get('source_config')!r}"
    )
if manifest.get("action_layout") != "multibinary_7":
    raise ValueError(
        f"Expected action_layout='multibinary_7', got {manifest.get('action_layout')!r}"
    )
if manifest.get("source_action_layout") != "deadly_corridor_joint_54":
    raise ValueError(
        "Expected source_action_layout='deadly_corridor_joint_54', got "
        f"{manifest.get('source_action_layout')!r}"
    )
if manifest.get("image_sequence_length") != expected_context_window:
    raise ValueError(
        "Converted image_sequence_length does not match the requested context window: "
        f"{manifest.get('image_sequence_length')} != {expected_context_window}"
    )
if manifest.get("max_episodes") != expected_max_episodes:
    raise ValueError(
        f"Converted max_episodes={manifest.get('max_episodes')} != {expected_max_episodes}"
    )
if manifest.get("latency_unit") != "raw_frames":
    raise ValueError(
        f"Expected latency_unit='raw_frames', got {manifest.get('latency_unit')!r}"
    )
if manifest.get("latency_raw_frames") != [6]:
    raise ValueError(
        f"Expected latency_raw_frames=[6], got {manifest.get('latency_raw_frames')!r}"
    )
if manifest.get("latency_env_steps") != [1.5]:
    raise ValueError(
        f"Expected latency_env_steps=[1.5], got {manifest.get('latency_env_steps')!r}"
    )
if set(prompt_map) != {"6"}:
    raise ValueError(
        f"Expected exactly raw-frame latency 6 in the prompt map, got {sorted(prompt_map)}"
    )
if prompt_map["6"].get("latency_raw_frames") != 6:
    raise ValueError(
        "Expected prompt-map latency_raw_frames=6, got "
        f"{prompt_map['6'].get('latency_raw_frames')!r}"
    )
PY
}

if [[ "${SKIP_ENV_SETUP}" != "true" ]]; then
  if [[ ! -f "${BENCHMARK_ROOT}/pyproject.toml" ]] \
    || [[ ! -d "${BENCHMARK_ROOT}/latency_bench" ]] \
    || [[ ! -f "${BENCHMARK_ROOT}/third_party/flappy-bird-gymnasium/setup.py" ]]; then
    echo "[deadly-wanoft-fixed] Invalid latency-sensitive-bench checkout: ${BENCHMARK_ROOT}" >&2
    echo "[deadly-wanoft-fixed] Pass --benchmark-root with a checkout whose submodules are initialized." >&2
    exit 1
  fi
  echo "[deadly-wanoft-fixed] Installing/updating env: ${CONDA_ENV_NAME}"
  BOOTSTRAP_ARGS=(
    bash
    examples/rl_games/install/bootstrap.sh
    --tier dev
    --model wan_oft
  )
  LATENCY_BENCH_ROOT="${BENCHMARK_ROOT}" "${BOOTSTRAP_ARGS[@]}"
fi

activate_conda_env
if [[ "${SKIP_CHECKPOINTS}" != "true" || "${SKIP_UPLOAD}" != "true" ]]; then
  ensure_hf_cli
fi

if [[ "${SKIP_CHECKPOINTS}" != "true" ]]; then
  echo "[deadly-wanoft-fixed] Downloading Wan base model checkpoint"
  hf download "${BASE_MODEL_REPO}" \
    --local-dir "${BASE_MODEL_DIR}"

  echo "[deadly-wanoft-fixed] Downloading WanOFT initialization checkpoint"
  hf download "${INIT_CHECKPOINT_REPO}" \
    --local-dir "${INIT_CHECKPOINT_DIR}"
fi

if [[ "${SKIP_CONVERT}" != "true" ]]; then
  echo "[deadly-wanoft-fixed] Converting ${DATASET_REPO}/${DATASET_CONFIG} into ${CONVERTED_DATA_DIR}"
  CONVERTER_ARGS=(
    python
    examples/rl_games/bash_scripts/gr00t/data_conversion/convert_deadly_corridor_history_to_starvla_lerobot.py
    --dataset-name "${DATASET_REPO}"
    --dataset-revision "${DATASET_REVISION}"
    --dataset-config-name "${DATASET_CONFIG}"
    --output-dir "${CONVERTED_DATA_DIR}"
    --max-episodes "${MAX_EPISODES}"
    --image-sequence-length "${CONTEXT_WINDOW}"
    --force
  )
  if [[ -n "${DATASET_CACHE_DIR}" ]]; then
    CONVERTER_ARGS+=(--cache-dir "${DATASET_CACHE_DIR}")
  fi
  "${CONVERTER_ARGS[@]}"
fi

if [[ "${SKIP_TRAIN}" != "true" ]]; then
  validate_converted_dataset
  echo "[deadly-wanoft-fixed] Training raw-frame latency=${LATENCY_RAW_FRAMES} with core eval disabled run_id=${RUN_ID}"
  export WANDB_ENTITY="${WANDB_ENTITY_VALUE}"
  export WANDB_PROJECT="${WANDB_PROJECT_VALUE}"

  TRAIN_OVERRIDES=(
    "paths.run_root_dir=${RUN_ROOT_DIR}"
    "paths.base_model_dir=${BASE_MODEL_DIR}"
    "initialization.checkpoint_local_dir=${INIT_CHECKPOINT_DIR}"
    "dataset.source_hf=${DATASET_REPO}"
    "dataset.config_name=${DATASET_CONFIG}"
    "dataset.max_episodes=${MAX_EPISODES}"
    "rl_games.env_eval.enabled=false"
    "rl_games.env_eval.mid_train.enabled=false"
    "rl_games.env_eval.post_train.enabled=false"
  )
  if [[ -n "${DATASET_CACHE_DIR}" ]]; then
    TRAIN_OVERRIDES+=("paths.dataset_cache_dir=${DATASET_CACHE_DIR}")
  fi

  CONTEXT_WINDOW="${CONTEXT_WINDOW}" \
  MAX_EPISODES="${MAX_EPISODES}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
  PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
  DATASET_LOCAL_DIR="${CONVERTED_DATA_ROOT}" \
  RUN_ID="${RUN_ID}" \
  PROMPT_MAP_PATH="${PROMPT_MAP_PATH}" \
    bash commands/wanoft/train_deadly_corridor_wan_oft.sh 6 "${TRAIN_OVERRIDES[@]}"
fi

if [[ "${SKIP_UPLOAD}" != "true" ]]; then
  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[deadly-wanoft-fixed] Training output directory does not exist: ${RUN_DIR}" >&2
    exit 1
  fi
  echo "[deadly-wanoft-fixed] Uploading ${RUN_DIR} to ${UPLOAD_REPO}:${UPLOAD_PATH_IN_REPO}"
  hf upload "${UPLOAD_REPO}" "${RUN_DIR}" "${UPLOAD_PATH_IN_REPO}" \
    --exclude "wandb/**" \
    --repo-type model
fi

echo "[deadly-wanoft-fixed] Complete: raw-frame latency=${LATENCY_RAW_FRAMES} core_eval=false run_id=${RUN_ID}"
