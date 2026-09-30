#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_ROOT="configs/train/memory/small_model/inverted_pendulum/latency_024"
FRAME_ROOT="${CONFIG_ROOT}/frame_stack"
MANIFEST="${CONFIG_ROOT}/manifest.tsv"
EVENT_LOG="${CONFIG_ROOT}/runner_events.jsonl"
TRAIN_DIR="results/checkpoints/memory/small_model/inverted_pendulum"
OUTPUT_ROOT="outputs/memory/small_model/inverted_pendulum/latency_024"
TRAIN_UPLOAD_REPO_ID="${TRAIN_UPLOAD_REPO_ID:-latency-sensitive-bench/paper-experiment-models}"
TRAIN_UPLOAD_REPO_TYPE="${TRAIN_UPLOAD_REPO_TYPE:-model}"
EVAL_UPLOAD_REPO_ID="${EVAL_UPLOAD_REPO_ID:-latency-sensitive-bench/paper-experiment-evals}"
EVAL_UPLOAD_REPO_TYPE="${EVAL_UPLOAD_REPO_TYPE:-dataset}"
UPLOAD_PREFIX="${UPLOAD_PREFIX:-memory/small_model/inverted_pendulum/latency_024_1m_fs1_eval}"
GPUS="${GPUS:-0}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-2}"
MAX_UPLOAD_ATTEMPTS="${MAX_UPLOAD_ATTEMPTS:-3}"
MATCH="${MATCH:-}"
DRY_RUN="${DRY_RUN:-0}"

if (($#)) && [[ "$1" =~ ^(train|eval|upload|all|generate)$ ]]; then
  STAGE="$1"
  shift
else
  STAGE="${STAGE:-train}"
fi

if (($#)); then
  SEEDS=("$@")
else
  SEEDS=(10 11 12 13 14)
fi

latency_labels() {
  case "${LATENCY:-all}" in
    0|l0|L0) echo "fixed_l0" ;;
    2|l2|L2) echo "fixed_l2" ;;
    4|l4|L4) echo "fixed_l4" ;;
    all) echo "fixed_l0 fixed_l2 fixed_l4" ;;
    *)
      echo "LATENCY must be one of: 0, 2, 4, all" >&2
      exit 2
      ;;
  esac
}

latency_frames() {
  case "$1" in
    fixed_l0) echo 0 ;;
    fixed_l2) echo 2 ;;
    fixed_l4) echo 4 ;;
    *) echo "Unknown latency label: $1" >&2; exit 2 ;;
  esac
}

latency_ms() {
  case "$1" in
    fixed_l0) echo "0.0" ;;
    fixed_l2) echo "80.0" ;;
    fixed_l4) echo "160.0" ;;
    *) echo "Unknown latency label: $1" >&2; exit 2 ;;
  esac
}

train_config_path() {
  local label="$1"
  local seed="$2"
  echo "${FRAME_ROOT}/${label}/fs1/seed_${seed}/train.yaml"
}

eval_config_path() {
  local label="$1"
  local seed="$2"
  echo "${FRAME_ROOT}/${label}/fs1/seed_${seed}/latency_bench_eval.yaml"
}

run_name() {
  local label="$1"
  local seed="$2"
  echo "inverted_pendulum_frame_stack_${label}_fs1_seed${seed}"
}

write_eval_config() {
  local label="$1"
  local seed="$2"
  local frames ms run eval_cfg
  frames="$(latency_frames "${label}")"
  ms="$(latency_ms "${label}")"
  run="$(run_name "${label}" "${seed}")"
  eval_cfg="$(eval_config_path "${label}" "${seed}")"
  mkdir -p "$(dirname "${eval_cfg}")"
  cat > "${eval_cfg}" <<EOF
__base__: ../../../../base_train.yaml

experiment:
  name: ${run}
  seed: ${seed}

backend:
  run_mode: eval

latency:
  fixed_latency_ms: ${ms}

policy:
  type: gymnasium_sf
  checkpoint_kind: latest
  device: cuda
  deterministic: true

evaluation:
  eval_episodes: 20
  eval_parallel_envs: 1
  eval_max_steps: 1000
  eval_deterministic: true
  eval_suites:
    fixed:
      - [${frames}]
    normal: []
    uniform: []

logging:
  output_dir: ${OUTPUT_ROOT}/latency_bench/frame_stack/${label}/fs1/seed_${seed}
  video:
    enabled: false
    num_bins: 1
  save_step_records: true
  save_action_records: true
  save_latency_records: true
  wandb_group: inverted-pendulum-fs1-${label}
  wandb_tags: [memory_small_model, inverted-pendulum, frame_stack, fixed, ${label}, fs1, seed${seed}, eval]
  wandb_name: inverted-pendulum-fs1-${label}-s${seed}-eval
EOF
}

write_manifest() {
  mkdir -p "$(dirname "${MANIFEST}")"
  cat > "${MANIFEST}" <<'EOF'
schema_version	job_id	env	experiment_family	sweep_name	latency_label	seed	frame_stack	obs_fps	observation_stride	train_config	eval_kind	eval_config	train_command	eval_command	run_name	wandb_group	wandb_name	train_dir	checkpoint_dir	train_output_dir	eval_output_dir	train_artifact_dir	eval_artifact_dir	train_upload_repo_id	train_upload_repo_type	eval_upload_repo_id	eval_upload_repo_type	train_artifact_repo_path	eval_artifact_repo_path	latency_kind	latency_fixed_frames	latency_mean_frames	latency_std_frames	latency_min_frames	latency_max_frames
EOF

  local label seed frames job run train_cfg eval_cfg train_output eval_output train_artifact eval_artifact
  for label in $(latency_labels); do
    frames="$(latency_frames "${label}")"
    for seed in "${SEEDS[@]}"; do
      write_eval_config "${label}" "${seed}"
      run="$(run_name "${label}" "${seed}")"
      job="memory_small_model:inverted_pendulum:frame_stack:${label}:fs1:obs25:stride1:seed${seed}"
      train_cfg="$(train_config_path "${label}" "${seed}")"
      eval_cfg="$(eval_config_path "${label}" "${seed}")"
      train_output="${OUTPUT_ROOT}/train/frame_stack/${label}/fs1/seed_${seed}"
      eval_output="${OUTPUT_ROOT}/latency_bench/frame_stack/${label}/fs1/seed_${seed}"
      train_artifact="${TRAIN_DIR}/${run}"
      eval_artifact="${eval_output}"
      local row=(
        v2
        "${job}"
        inverted_pendulum
        memory_small_model
        frame_stack
        "${label}"
        "${seed}"
        1
        25
        1
        "${train_cfg}"
        latency_bench
        "${eval_cfg}"
        "python -m training.train --config ${train_cfg}"
        "python -m latency_bench.run --config ${eval_cfg}"
        "${run}"
        "inverted-pendulum-fs1-${label}"
        "inverted-pendulum-fs1-${label}-s${seed}"
        "${TRAIN_DIR}"
        "${train_artifact}/checkpoint_p0"
        "${train_output}"
        "${eval_output}"
        "${train_artifact}"
        "${eval_artifact}"
        "${TRAIN_UPLOAD_REPO_ID}"
        "${TRAIN_UPLOAD_REPO_TYPE}"
        "${EVAL_UPLOAD_REPO_ID}"
        "${EVAL_UPLOAD_REPO_TYPE}"
        "${UPLOAD_PREFIX}/train/${job}"
        "${UPLOAD_PREFIX}/eval/${job}"
        fixed
        "${frames}"
        ""
        ""
        ""
        ""
      )
      (IFS=$'\t'; echo "${row[*]}") >> "${MANIFEST}"
    done
  done
}

run_manifest_stage() {
  local stage="$1"
  write_manifest
  local args=(--manifest "${MANIFEST}" --stage "${stage}")
  case "${stage}" in
    train|eval|all)
      args+=(--gpus "${GPUS}")
      ;;
    upload)
      args+=(--gpus "${GPUS}" --upload-workers "${UPLOAD_WORKERS}" --max-upload-attempts "${MAX_UPLOAD_ATTEMPTS}")
      ;;
  esac
  if [[ -n "${MATCH}" ]]; then
    args+=(--match "${MATCH}")
  fi
  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    args+=(--dry-run)
  fi
  python -m scripts.experiment_runner.run_manifest_jobs "${args[@]}"
}

train_inverted_pendulum_latency_0() {
  LATENCY=0 run_manifest_stage train
}

train_inverted_pendulum_latency_2() {
  LATENCY=2 run_manifest_stage train
}

train_inverted_pendulum_latency_4() {
  LATENCY=4 run_manifest_stage train
}

case "${STAGE}" in
  generate)
    write_manifest
    echo "Wrote ${MANIFEST}"
    ;;
  train|eval|upload|all)
    run_manifest_stage "${STAGE}"
    ;;
  *)
    echo "Usage: $0 [generate|train|eval|upload|all] [seed ...]" >&2
    echo "Set LATENCY=0,2,4,all; GPUS=0,1; HF_TOKEN=... for upload." >&2
    exit 2
    ;;
esac
