#!/usr/bin/env bash
set -euo pipefail

# Run from the latency-sensitive-bench repo root on a GPU machine:
#   conda activate latency
#   export HF_TOKEN=hf_...
#   bash scripts/experiments/memory/rollout_inverted_pendulum_latency024.sh l0
#   bash scripts/experiments/memory/rollout_inverted_pendulum_latency024.sh all
#
# Source it if you want functions:
#   source scripts/experiments/memory/rollout_inverted_pendulum_latency024.sh
#   rollout_inverted_pendulum_l0
#   rollout_inverted_pendulum_l2
#   rollout_inverted_pendulum_l4
#
# Defaults can be overridden:
#   MAX_EPISODES=200
#   MAX_STEPS=7200
#   NUM_ENVS=32
#   REPLAY_NUM_ENVS=32
#   DEVICE=cuda
#   FILTER_PRESET=none
#   EPISODE_RETURN_GT=<override acceptance gate, or none to disable>
#   MAX_ATTEMPTS=<optional maximum rollout attempts>
#   HF_ROLLOUT_REPO=latency-sensitive-bench/inverted_pendulum_200ep
#   PUSH_TO_HUB=1

export HF_MODEL_REPO="${HF_MODEL_REPO:-latency-sensitive-bench/paper-experiment-models}"
export HF_ROLLOUT_REPO="${HF_ROLLOUT_REPO:-latency-sensitive-bench/inverted_pendulum_200ep}"
export SMALL_MODEL_CKPT_ROOT="${SMALL_MODEL_CKPT_ROOT:-/mnt/local/talha/inverted_pendulum_latency024_small_model_ckpts}"
export ROLLOUT_DATA_ROOT="${ROLLOUT_DATA_ROOT:-/mnt/local/talha/inverted_pendulum_latency024_rollout_datasets}"

_ip_need_repo_root() {
  if [[ ! -f scripts/rollout_data/export_sf_teacher_dataset.py ]]; then
    echo "[error] run this from the latency-sensitive-bench repo root." >&2
    return 1
  fi
}

_ip_need_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[error] missing command: ${cmd}" >&2
    return 1
  fi
}

_ip_setup_cuda_env() {
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

_ip_latency_label() {
  case "$1" in
    0|l0|L0|fixed_l0) echo "fixed_l0" ;;
    2|l2|L2|fixed_l2) echo "fixed_l2" ;;
    4|l4|L4|fixed_l4) echo "fixed_l4" ;;
    *)
      echo "[error] latency must be one of 0, 2, 4" >&2
      return 2
      ;;
  esac
}

_ip_latency_frames() {
  case "$(_ip_latency_label "$1")" in
    fixed_l0) echo 0 ;;
    fixed_l2) echo 2 ;;
    fixed_l4) echo 4 ;;
  esac
}

_ip_latency_ms() {
  case "$(_ip_latency_label "$1")" in
    fixed_l0) echo "0.0" ;;
    fixed_l2) echo "80.0" ;;
    fixed_l4) echo "160.0" ;;
  esac
}

_ip_steps_label() {
  case "${MAX_STEPS:-7200}" in
    7200) echo "7k2steps" ;;
    3600) echo "3k6steps" ;;
    *) echo "${MAX_STEPS:-7200}steps" ;;
  esac
}

_ip_default_episode_return_gt() {
  case "$(_ip_latency_label "$1")" in
    fixed_l0) echo 7000 ;;
    fixed_l2) echo 85 ;;
    fixed_l4) echo 24 ;;
  esac
}

_ip_best_remote_ckpt() {
  local latency="$1"
  local label
  label="$(_ip_latency_label "${latency}")" || return 1

  local api_url="https://huggingface.co/api/models/${HF_MODEL_REPO}/tree/main/memory/small_model/inverted_pendulum?recursive=true"

  curl -fsSL "${api_url}" |
    LATENCY_LABEL="${label}" python3 -c '
import json
import os
import re
import sys

label = os.environ["LATENCY_LABEL"]
pattern = re.compile(
    r"^memory/small_model/inverted_pendulum/latency_024_1m_fs1_eval/train/"
    r"memory_small_model:inverted_pendulum:frame_stack:" + re.escape(label) +
    r":fs1:obs25:stride1:seed10/checkpoint_p0/"
    r"best_.*reward_(-?\d+(?:\.\d+)?)\.pth$"
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

download_best_inverted_pendulum_ckpt() {
  local latency="$1"

  _ip_need_cmd curl || return 1
  _ip_need_cmd hf || return 1

  local label remote_ckpt remote_exp_dir remote_config local_root ckpt ckpt_name
  label="$(_ip_latency_label "${latency}")" || return 1
  remote_ckpt="$(_ip_best_remote_ckpt "${latency}")" || {
    echo "[error] no HF checkpoint found for inverted_pendulum ${label} in ${HF_MODEL_REPO}" >&2
    return 1
  }
  remote_exp_dir="${remote_ckpt%/checkpoint_p0/*}"
  remote_config="${remote_exp_dir}/config.json"
  local_root="${SMALL_MODEL_CKPT_ROOT}/${label}"
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

  echo "[selected] ${ckpt}" >&2
  printf '%s\n' "${ckpt}"
}

_ip_write_latency_config() {
  local latency="$1"
  local label ms path
  label="$(_ip_latency_label "${latency}")" || return 1
  ms="$(_ip_latency_ms "${latency}")" || return 1
  path="${ROLLOUT_DATA_ROOT}/_latency_configs/${label}.yaml"
  mkdir -p "$(dirname "${path}")"
  cat > "${path}" <<EOF
latency:
  method: fixed
  fixed_latency_ms: ${ms}
  add_latency_info: false
EOF
  printf '%s\n' "${path}"
}

_ip_rollout_one() {
  local latency="$1"

  _ip_need_repo_root || return 1
  _ip_setup_cuda_env

  local label frames ckpt latency_config config_name out_dir steps_label gym_make_kwargs episode_return_gt
  label="$(_ip_latency_label "${latency}")" || return 1
  frames="$(_ip_latency_frames "${latency}")" || return 1
  ckpt="$(download_best_inverted_pendulum_ckpt "${latency}")" || return 1
  latency_config="$(_ip_write_latency_config "${latency}")" || return 1
  steps_label="$(_ip_steps_label)"
  config_name="inverted_pendulum_fixed_latency_${frames}_${MAX_EPISODES:-200}ep_${steps_label}"
  out_dir="${ROLLOUT_DATA_ROOT}/${config_name}"
  gym_make_kwargs="{\"base_make_kwargs\":{\"max_episode_steps\":${MAX_STEPS:-7200}}}"
  episode_return_gt="${EPISODE_RETURN_GT:-$(_ip_default_episode_return_gt "${latency}")}"

  echo "[run] inverted_pendulum ${label}"
  echo "[ckpt] ${ckpt}"
  echo "[latency] ${latency_config}"
  echo "[out]  ${out_dir}"
  echo "[hf]   ${HF_ROLLOUT_REPO}/${config_name} (PUSH_TO_HUB=${PUSH_TO_HUB:-1})"
  echo "[steps] MAX_EPISODES=${MAX_EPISODES:-200} MAX_STEPS=${MAX_STEPS:-7200}"
  echo "[gate] FILTER_PRESET=${FILTER_PRESET:-none} EPISODE_RETURN_GT=${episode_return_gt}"

  local export_wrapper
  export_wrapper='
import os
import sys

# Gymnasium plugin discovery imports shimmy/dm_control. On headless machines,
# importing those plugins with MUJOCO_GL=egl can fail before our own task starts.
# Load Gymnasium plugins first, then set MuJoCo EGL for the actual rgb_array
# renderer used by InvertedPendulum-v4 replay.
os.environ.pop("MUJOCO_GL", None)
os.environ.pop("PYOPENGL_PLATFORM", None)
import gymnasium  # noqa: F401

os.environ["MUJOCO_GL"] = os.environ.get("ROLLOUT_MUJOCO_GL", "egl")

from scripts.rollout_data.export_sf_teacher_dataset import main

raise SystemExit(main(sys.argv[1:]))
'

  local cmd=(
    python -c "${export_wrapper}" rollout_export
    --checkpoint-root "${ckpt}"
    --output-dir "${out_dir}"
    --latency-config "${latency_config}"
    --gym-make-kwargs-json "${gym_make_kwargs}"
    --max-episodes "${MAX_EPISODES:-200}"
    --max-steps-per-episode "${MAX_STEPS:-7200}"
    --val-fraction "${VAL_FRACTION:-0.01}"
    --num-envs "${NUM_ENVS:-32}"
    --replay-num-envs "${REPLAY_NUM_ENVS:-32}"
    --device "${DEVICE:-cuda}"
    --context-window "${CONTEXT_WINDOW:-1}"
    --image-transform "${IMAGE_TRANSFORM:-raw_rgb}"
    --image-writer-workers "${IMAGE_WRITER_WORKERS:-16}"
    --filter-preset "${FILTER_PRESET:-none}"
  )

  if [[ -n "${MAX_ATTEMPTS:-}" ]]; then
    cmd+=(--max-attempts "${MAX_ATTEMPTS}")
  fi
  if [[ -n "${EXPORT_IMAGE_SIZE:-}" ]]; then
    cmd+=(--export-image-size "${EXPORT_IMAGE_SIZE}")
  fi
  if [[ -n "${episode_return_gt}" && "${episode_return_gt}" != "none" && "${episode_return_gt}" != "NONE" ]]; then
    cmd+=(--episode-return-gt "${episode_return_gt}")
  fi
  if [[ "${PROBE_ONLY:-0}" == "1" || "${PROBE_ONLY:-0}" == "true" ]]; then
    cmd+=(--probe-only)
  fi
  if [[ "${PUSH_TO_HUB:-1}" == "1" || "${PUSH_TO_HUB:-1}" == "true" ]]; then
    cmd+=(--push-to-hub --hf-repo-id "${HF_ROLLOUT_REPO}" --hf-token "${HF_TOKEN:-}")
    if [[ "${HF_PRIVATE:-0}" == "1" || "${HF_PRIVATE:-0}" == "true" ]]; then
      cmd+=(--hf-private)
    fi
    cmd+=(--hf-config-name "${config_name}")
  fi

  "${cmd[@]}"
}

rollout_inverted_pendulum_l0() { _ip_rollout_one 0; }
rollout_inverted_pendulum_l2() { _ip_rollout_one 2; }
rollout_inverted_pendulum_l4() { _ip_rollout_one 4; }

rollout_inverted_pendulum_all() {
  rollout_inverted_pendulum_l0
  rollout_inverted_pendulum_l2
  rollout_inverted_pendulum_l4
}

print_inverted_pendulum_rollout_help() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/memory/rollout_inverted_pendulum_latency024.sh l0
  bash scripts/experiments/memory/rollout_inverted_pendulum_latency024.sh l2
  bash scripts/experiments/memory/rollout_inverted_pendulum_latency024.sh l4
  bash scripts/experiments/memory/rollout_inverted_pendulum_latency024.sh all

Functions when sourced:
  rollout_inverted_pendulum_l0
  rollout_inverted_pendulum_l2
  rollout_inverted_pendulum_l4
  rollout_inverted_pendulum_all

Defaults:
  MAX_EPISODES=200
  MAX_STEPS=7200
  NUM_ENVS=32
  REPLAY_NUM_ENVS=32
  DEVICE=cuda
  FILTER_PRESET=none
  EPISODE_RETURN_GT defaults by latency: l0 > 7000, l2 > 85, l4 > 24
  Set EPISODE_RETURN_GT=none to disable, or EPISODE_RETURN_GT=<number> to override.
  MAX_ATTEMPTS=<optional maximum total rollout attempts>
  HF_ROLLOUT_REPO=latency-sensitive-bench/inverted_pendulum_200ep
  PUSH_TO_HUB=1

The script downloads the best seed10 small-model checkpoint for each latency
from latency-sensitive-bench/paper-experiment-models, forces the Gymnasium
InvertedPendulum-v4 episode cap to MAX_STEPS, exports raw-RGB rollout data, and
uploads it as a Hugging Face dataset config when HF_ROLLOUT_REPO is set.
EOF
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-help}" in
    l0|0|fixed_l0) rollout_inverted_pendulum_l0 ;;
    l2|2|fixed_l2) rollout_inverted_pendulum_l2 ;;
    l4|4|fixed_l4) rollout_inverted_pendulum_l4 ;;
    all) rollout_inverted_pendulum_all ;;
    help|-h|--help) print_inverted_pendulum_rollout_help ;;
    *)
      echo "[error] unknown target: ${1}" >&2
      print_inverted_pendulum_rollout_help >&2
      exit 2
      ;;
  esac
fi
