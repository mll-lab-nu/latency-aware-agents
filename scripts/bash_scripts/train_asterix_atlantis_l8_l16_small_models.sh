#!/usr/bin/env bash
set -euo pipefail

# Generate and run Sample Factory small-model training for Asterix/Atlantis
# fixed latency 8 and 16 raw env frames.
#
# Defaults match the existing memory/small_model Asterix/Atlantis latency_024
# setup: fs4, obs_fps=15, 10M env steps, seeds 10..14.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-latency}"
ENVS_CSV="${ENVS:-asterix,atlantis}"
SEEDS="${SEEDS:-10,11,12,13,14}"
GPUS="${GPUS:-}"
LATENCY_SET_NAME="${LATENCY_SET_NAME:-latency_8_16}"
TRAIN_STEPS="${TRAIN_STEPS:-10000000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-3600}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-2}"
MAX_UPLOAD_ATTEMPTS="${MAX_UPLOAD_ATTEMPTS:-3}"
MATCH="${MATCH:-}"
DRY_RUN=false

usage() {
  cat >&2 <<'EOF'
Usage:
  train_asterix_atlantis_l8_l16_small_models.sh generate
  train_asterix_atlantis_l8_l16_small_models.sh train --gpus 0,1
  train_asterix_atlantis_l8_l16_small_models.sh eval --gpus 0,1
  train_asterix_atlantis_l8_l16_small_models.sh upload
  train_asterix_atlantis_l8_l16_small_models.sh all --gpus 0,1

Useful overrides:
  CONDA_ENV=latency
  ENVS=asterix,atlantis
  SEEDS=10,11,12,13,14
  GPUS=0,1
  MATCH=fixed_l8
  DRY_RUN=1
  UPLOAD_WORKERS=2
  MAX_UPLOAD_ATTEMPTS=3

Outputs:
  configs/experiments/memory/small_model/{asterix,atlantis}_latency_8_16.yaml
  configs/train/memory/small_model/{asterix,atlantis}/latency_8_16/manifest.tsv
  results/checkpoints/memory/small_model/{asterix,atlantis}/...
  outputs/memory/small_model/{asterix,atlantis}/latency_8_16/...

Latency meaning:
  l8  = 8 raw frames at 60 FPS  = 133.333 ms
  l16 = 16 raw frames at 60 FPS = 266.667 ms
EOF
}

python_cmd() {
  if [[ -n "${CONDA_ENV}" ]]; then
    conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
  else
    python "$@"
  fi
}

csv_to_words() {
  local value="$1"
  printf '%s\n' "${value//,/ }"
}

spec_path_for_env() {
  local env_name="$1"
  echo "configs/experiments/memory/small_model/${env_name}_latency_8_16.yaml"
}

manifest_path_for_env() {
  local env_name="$1"
  echo "configs/train/memory/small_model/${env_name}/${LATENCY_SET_NAME}/manifest.tsv"
}

write_spec() {
  local env_name="$1"
  local spec_path base_path
  spec_path="$(spec_path_for_env "${env_name}")"
  base_path="../../../base/${env_name}.yaml"
  mkdir -p "$(dirname "${spec_path}")"

  cat > "${spec_path}" <<EOF
__base__: ${base_path}
experiment_family: memory_small_model
generated_root: configs/train/memory/small_model/${env_name}/${LATENCY_SET_NAME}
train_dir: results/checkpoints/memory/small_model/${env_name}
output_root: outputs/memory/small_model/${env_name}/${LATENCY_SET_NAME}
device: cuda
restart_behavior: resume
train_upload_repo_id: latency-sensitive-bench/paper-experiment-models
train_upload_repo_type: model
train_upload_path_prefix: memory/small_model/${env_name}/${LATENCY_SET_NAME}_10m_fs4_1min_eval
eval_upload_repo_id: latency-sensitive-bench/paper-experiment-evals
eval_upload_repo_type: dataset
eval_upload_path_prefix: memory/small_model/${env_name}/${LATENCY_SET_NAME}_10m_fs4_1min_eval
seeds:
  - 10
  - 11
  - 12
  - 13
  - 14

evaluation:
  eval_episodes: ${EVAL_EPISODES}
  eval_parallel_envs: 1
  eval_max_steps: ${EVAL_MAX_STEPS}

training:
  train_for_env_steps: ${TRAIN_STEPS}

latency_settings:
  fixed_l8:
    kind: fixed
    frames: 8
    eval_suites:
      fixed:
        - [8]
      normal: []
      uniform: []
  fixed_l16:
    kind: fixed
    frames: 16
    eval_suites:
      fixed:
        - [16]
      normal: []
      uniform: []

sweeps:
  frame_stack:
    frame_stack:
      - 4
    obs_fps:
      - 15
EOF
}

generate_configs() {
  local env_name spec_path
  for env_name in $(csv_to_words "${ENVS_CSV}"); do
    case "${env_name}" in
      asterix|atlantis) ;;
      *) echo "Unsupported env=${env_name}; expected asterix or atlantis" >&2; exit 2 ;;
    esac
    write_spec "${env_name}"
    spec_path="$(spec_path_for_env "${env_name}")"
    echo "==> generating ${spec_path}"
    python_cmd scripts/sweeps/generate_sweep_configs.py --spec "${spec_path}" --seeds "${SEEDS}"
  done
}

run_stage() {
  local stage="$1"
  local env_name manifest args=()
  case "${stage}" in
    train|eval)
      if [[ -z "${GPUS}" ]]; then
        echo "--gpus is required for ${stage}; pass --gpus 0 or set GPUS=0" >&2
        exit 2
      fi
      args=(--stage "${stage}" --gpus "${GPUS}")
      ;;
    upload)
      args=(--stage upload --gpus "${GPUS:-0}" --upload-workers "${UPLOAD_WORKERS}" --max-upload-attempts "${MAX_UPLOAD_ATTEMPTS}")
      ;;
    *) echo "Unknown stage=${stage}" >&2; exit 2 ;;
  esac
  if [[ -n "${MATCH}" ]]; then
    args+=(--match "${MATCH}")
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    args+=(--dry-run)
  fi
  for env_name in $(csv_to_words "${ENVS_CSV}"); do
    manifest="$(manifest_path_for_env "${env_name}")"
    if [[ ! -f "${manifest}" ]]; then
      echo "Missing manifest ${manifest}; run generate first." >&2
      exit 2
    fi
    echo "==> ${stage} ${manifest}"
    python_cmd -m scripts.experiment_runner.run_manifest_jobs --manifest "${manifest}" "${args[@]}"
  done
}

command="${1:-}"
if [[ -z "${command}" || "${command}" == "-h" || "${command}" == "--help" ]]; then
  usage
  exit $([[ -z "${command}" ]] && echo 2 || echo 0)
fi
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --envs) ENVS_CSV="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --match) MATCH="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) usage; exit 2 ;;
  esac
done
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=true
fi

case "${command}" in
  generate)
    generate_configs
    ;;
  train)
    run_stage train
    ;;
  eval)
    run_stage eval
    ;;
  upload)
    run_stage upload
    ;;
  all)
    generate_configs
    run_stage train
    run_stage eval
    run_stage upload
    ;;
  *)
    usage
    exit 2
    ;;
esac
