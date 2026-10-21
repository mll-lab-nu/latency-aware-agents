#!/usr/bin/env bash
set -euo pipefail

# Generate/upload fixed-latency rollout datasets for the latency-0/2/4
# small-model teachers used to build mixed-latency data.
#
# New machine usage:
#   cd /path/to/latency-sensitive-bench
#   conda activate latency
#   export HF_TOKEN=hf_...
#   bash scripts/bash_scripts/rollout_data/rollout_latency024_atari.sh all
#
# Or source and call individual functions:
#   source scripts/bash_scripts/rollout_data/rollout_latency024_atari.sh
#   rollout_asterix_l0
#   rollout_asterix_l2
#   rollout_asterix_l4
#   rollout_atlantis_l0
#   rollout_atlantis_l2
#   rollout_atlantis_l4
#
# Common overrides:
#   MAX_EPISODES=200
#   MAX_STEPS=7200
#   NUM_ENVS=64
#   REPLAY_NUM_ENVS=64
#   DEVICE=cuda
#   FILTER_PRESET=none
#   EPISODE_RETURN_GT=<override per-task gate>
#   HF_MODEL_REPO=latency-sensitive-bench/paper-experiment-models
#   HF_ROLLOUT_REPO=<override one shared upload repo>
#
# Default teacher seeds are one selected small model per env/latency:
#   asterix:  l0=s11, l2=s14, l4=s14
#   atlantis: l0=s12, l2=s14, l4=s10
#   defend:   l2=s0,  l4=s0

export HF_MODEL_REPO="${HF_MODEL_REPO:-latency-sensitive-bench/paper-experiment-models}"
export HF_ROLLOUT_REPO="${HF_ROLLOUT_REPO:-MLL-Lab/LAGEN-datasets}"
export SMALL_MODEL_CKPT_ROOT="${SMALL_MODEL_CKPT_ROOT:-/mnt/local/talha/latency024_small_model_ckpts}"
export ROLLOUT_DATA_ROOT="${ROLLOUT_DATA_ROOT:-/mnt/local/talha/latency024_rollout_datasets}"

need_repo_root() {
  if [[ ! -f scripts/bash_scripts/run_sf_teacher_dataset_export.sh ]]; then
    echo "[error] run this from the latency-sensitive-bench repo root." >&2
    return 1
  fi
}

need_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[error] missing command: ${cmd}" >&2
    return 1
  fi
}

setup_cuda_env() {
  if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
      export CUDA_HOME="${CONDA_PREFIX}"
    elif [[ -d /usr/local/cuda ]]; then
      export CUDA_HOME=/usr/local/cuda
    fi
  fi

  if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi

  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;9.0;12.0}"
}

tag_for_env() {
  case "$1" in
    asterix|atlantis) printf '%s\n' "10m_fs4_1min_eval" ;;
    *)
      echo "[error] unknown env: $1" >&2
      return 2
      ;;
  esac
}

model_family_for_env_latency() {
  local env_name="$1"
  local latency="$2"
  local tag
  tag="$(tag_for_env "${env_name}")" || return 1

  if [[ "${latency}" == "0" ]]; then
    printf '%s\n' "zero_latency_${tag}"
    return 0
  fi

  printf '%s\n' "latency_024_${tag}"
}

selected_seed_for_env_latency() {
  local env_name="$1"
  local latency="$2"
  local override_key

  override_key="$(printf '%s_L%s_SEED' "${env_name}" "${latency}" | tr '[:lower:]' '[:upper:]')"
  if [[ -n "${!override_key:-}" ]]; then
    printf '%s\n' "${!override_key}"
    return 0
  fi

  case "${env_name}:${latency}" in
    asterix:0) printf '%s\n' 11 ;;
    asterix:2) printf '%s\n' 14 ;;
    asterix:4) printf '%s\n' 14 ;;
    atlantis:0) printf '%s\n' 12 ;;
    atlantis:2) printf '%s\n' 14 ;;
    atlantis:4) printf '%s\n' 10 ;;
    *)
      echo "[error] no selected seed for ${env_name} latency ${latency}" >&2
      return 1
      ;;
  esac
}

default_episode_return_gt_for_env_latency() {
  local env_name="$1"
  local latency="$2"
  case "${env_name}:${latency}" in
    asterix:0) printf '%s\n' 6000 ;;
    asterix:2) printf '%s\n' 6000 ;;
    asterix:4) printf '%s\n' 6000 ;;
    atlantis:0) printf '%s\n' 30000 ;;
    atlantis:2) printf '%s\n' 30000 ;;
    atlantis:4) printf '%s\n' 30000 ;;
    *)
      echo "[error] no default return gate for ${env_name} latency ${latency}" >&2
      return 1
      ;;
  esac
}


best_remote_ckpt() {
  local env_name="$1"
  local latency="$2"
  local family seed
  family="$(model_family_for_env_latency "${env_name}" "${latency}")" || return 1
  seed="$(selected_seed_for_env_latency "${env_name}" "${latency}")" || return 1

  local api_url="https://huggingface.co/api/models/${HF_MODEL_REPO}/tree/main/memory/small_model/${env_name}?recursive=true"

  curl -fsSL "${api_url}" |
    ENV_NAME="${env_name}" LATENCY="${latency}" FAMILY="${family}" SEED="${seed}" python3 -c '
import json
import os
import re
import sys

env_name = os.environ["ENV_NAME"]
latency = os.environ["LATENCY"]
family = os.environ["FAMILY"]
seed = os.environ["SEED"]

pattern = re.compile(
    r"^memory/small_model/" + re.escape(env_name) + r"/" + re.escape(family)
    + r"/train/memory_small_model:" + re.escape(env_name)
    + r":frame_stack:fixed_l" + re.escape(latency)
    + r":fs4:obs[^:]+:stride4:seed" + re.escape(seed) + r"/checkpoint_p0/"
    + r"best_.*reward_(-?\d+(?:\.\d+)?)\.pth$"
)

rows = []
for item in json.load(sys.stdin):
    path = item.get("path", "")
    match = pattern.match(path)
    if match:
        rows.append((float(match.group(1)), path))

if not rows:
    raise SystemExit(1)

rows.sort(key=lambda item: item[0])
print(rows[-1][1])
'
}

download_best_latency024_ckpt() {
  local env_name="$1"
  local latency="$2"

  need_cmd curl || return 1
  need_cmd hf || return 1

  local remote_ckpt remote_exp_dir remote_config local_root ckpt_name ckpt
  remote_ckpt="$(best_remote_ckpt "${env_name}" "${latency}")" || {
    echo "[error] no HF checkpoint found for ${env_name} fixed_l${latency} in ${HF_MODEL_REPO}" >&2
    return 1
  }
  remote_exp_dir="${remote_ckpt%/checkpoint_p0/*}"
  remote_config="${remote_exp_dir}/config.json"
  local_root="${SMALL_MODEL_CKPT_ROOT}/${env_name}/fixed_l${latency}"
  mkdir -p "${local_root}"

  echo "[download] ${HF_MODEL_REPO}:${remote_config}" >&2
  hf download "${HF_MODEL_REPO}" "${remote_config}" \
    --repo-type model \
    --local-dir "${local_root}" >/dev/null

  echo "[download] ${HF_MODEL_REPO}:${remote_ckpt}" >&2
  hf download "${HF_MODEL_REPO}" "${remote_ckpt}" \
    --repo-type model \
    --local-dir "${local_root}" >/dev/null

  ckpt_name="$(basename "${remote_ckpt}")"
  ckpt="$(find "${local_root}" -type f -path "*/checkpoint_p0/${ckpt_name}" -print -quit)"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[error] downloaded checkpoint missing under ${local_root}" >&2
    find "${local_root}" -type f -name "*.pth" -print >&2
    return 1
  fi

  printf '%s\n' "${ckpt}"
}

rollout_latency024_one() {
  local env_name="$1"
  local latency="$2"

  need_repo_root || return 1
  setup_cuda_env

  local ckpt config_name out_dir steps_label hf_rollout_repo
  ckpt="$(download_best_latency024_ckpt "${env_name}" "${latency}")" || return 1
  hf_rollout_repo="${HF_ROLLOUT_REPO}"

  case "${MAX_STEPS:-7200}" in
    7200) steps_label="7k2steps" ;;
    3600) steps_label="3k6steps" ;;
    *) steps_label="${MAX_STEPS:-7200}steps" ;;
  esac

  config_name="${env_name}_fixed_latency_${latency}_${MAX_EPISODES:-200}ep_${steps_label}"
  out_dir="${ROLLOUT_DATA_ROOT}/${config_name}"
  episode_return_gt="${EPISODE_RETURN_GT:-$(default_episode_return_gt_for_env_latency "${env_name}" "${latency}")}"

  echo "[run] ${env_name} fixed_l${latency}"
  echo "[ckpt] ${ckpt}"
  echo "[out]  ${out_dir}"
  echo "[hf]   ${hf_rollout_repo}/${config_name}"
  echo "[gate] FILTER_PRESET=${FILTER_PRESET:-none} EPISODE_RETURN_GT=${episode_return_gt}"

  MAX_EPISODES="${MAX_EPISODES:-200}" \
  MAX_STEPS="${MAX_STEPS:-7200}" \
  NUM_ENVS="${NUM_ENVS:-64}" \
  REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS:-64}" \
  DEVICE="${DEVICE:-cuda}" \
  CONTEXT_WINDOW="${CONTEXT_WINDOW:-1}" \
  FILTER_PRESET="${FILTER_PRESET:-none}" \
  EPISODE_RETURN_GT="${episode_return_gt}" \
  IMAGE_WRITER_WORKERS="${IMAGE_WRITER_WORKERS:-16}" \
  HF_REPO_ID="${hf_rollout_repo}" \
  HF_CONFIG_NAME="task-transfer--${config_name}" \
  HF_DATA_DIR="$(if [[ "${latency}" == "0" ]]; then echo zero-latency; else echo latency-aware; fi)/${env_name//_/-}/${env_name}_200ep/${config_name}" \
  HF_PRIVATE="${HF_PRIVATE:-0}" \
    bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
      "${ckpt}" \
      "${out_dir}"
}

rollout_asterix_l0() { rollout_latency024_one asterix 0; }
rollout_asterix_l2() { rollout_latency024_one asterix 2; }
rollout_asterix_l4() { rollout_latency024_one asterix 4; }

rollout_atlantis_l0() { rollout_latency024_one atlantis 0; }
rollout_atlantis_l2() { rollout_latency024_one atlantis 2; }
rollout_atlantis_l4() { rollout_latency024_one atlantis 4; }


rollout_asterix_all() {
  rollout_asterix_l0
  rollout_asterix_l2
  rollout_asterix_l4
}

rollout_atlantis_all() {
  rollout_atlantis_l0
  rollout_atlantis_l2
  rollout_atlantis_l4
}


rollout_all_latency024() {
  rollout_asterix_all
  rollout_atlantis_all
}

print_help() {
  cat <<'EOF'
Usage:
  bash scripts/bash_scripts/rollout_data/rollout_latency024_atari.sh <target>

Targets:
  asterix-l0 | asterix-l2 | asterix-l4 | asterix | asterix-all
  atlantis-l0 | atlantis-l2 | atlantis-l4 | atlantis | atlantis-all
  all

Source mode:
  source scripts/bash_scripts/rollout_data/rollout_latency024_atari.sh
  rollout_asterix_l0
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-help}" in
    asterix-l0) rollout_asterix_l0 ;;
    asterix-l2) rollout_asterix_l2 ;;
    asterix-l4) rollout_asterix_l4 ;;
    asterix|asterix-all) rollout_asterix_all ;;
    atlantis-l0) rollout_atlantis_l0 ;;
    atlantis-l2) rollout_atlantis_l2 ;;
    atlantis-l4) rollout_atlantis_l4 ;;
    atlantis|atlantis-all) rollout_atlantis_all ;;
    all) rollout_all_latency024 ;;
    help|-h|--help) print_help ;;
    *)
      echo "[error] unknown target: ${1}" >&2
      print_help >&2
      exit 2
      ;;
  esac
fi
