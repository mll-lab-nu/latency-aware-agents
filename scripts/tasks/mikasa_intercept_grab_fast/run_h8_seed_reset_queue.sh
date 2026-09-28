#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/outputs/starvla/datasets/mikasa_intercept_grab_fast/conditioned-h8-seed-reset-20260922"
OUTPUT_ROOT="${REPO_ROOT}/outputs/starvla/runs/mikasa_intercept_grab_fast/h8-conditioned-seed-reset-20260922"
QUEUE_ROOT="${OUTPUT_ROOT}/queue"
PROFILE_SCRIPT="scripts/tasks/mikasa_intercept_grab_fast/run_profile_v2_h8.sh"
FIXED_SCRIPT="scripts/tasks/mikasa_intercept_grab_fast/run_fixed_v2_h8.sh"
H1_ROOT="/home/ubuntu/codex/mikasa-h1-latency-20260922/outputs/latency_rl/mikasa"

export QWENOFT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_openvla/bin/python
export QWENPI_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_pi05/bin/python
export QWENGROOT_PY=/home/ubuntu/miniconda3/envs/starvla_rl_games_gr00t/bin/python
export MIKASA_PY="${REPO_ROOT}/third_party/MIKASA-Robo/.venv/bin/python"
export UV=/home/ubuntu/.local/bin/uv
export MIKASA_H8_DATA_ROOT="${DATA_ROOT}"
export MIKASA_H8_OUTPUT_ROOT="${OUTPUT_ROOT}"
export MIKASA_H8_PROFILE_RUN_ID=h8-conditioned-seed-reset-20260922-profile
export MIKASA_H8_FIXED_RUN_ID=h8-conditioned-seed-reset-20260922-fixed-2

cd "${REPO_ROOT}"
set -a
. "${REPO_ROOT}/.env"
set +a
export WANDB_ENTITY=saberrr-zju
export WANDB_MODE=online
export WANDB_RESUME=never

test ! -e "${DATA_ROOT}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${QUEUE_ROOT}"
EVENTS="${QUEUE_ROOT}/events.tsv"

record() {
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" >> "${EVENTS}"
}

run_data_job() {
  local job="$1"
  local gpu="$2"
  case "${job}" in
    fixed-2)
      FIXED_COLLECT_GPU="${gpu}" bash "${FIXED_SCRIPT}" collect
      ;;
    profile-*)
      local model="${job#profile-}"
      PROFILE_MODELS="${model}" PROFILE_COLLECT_GPUS="${gpu}" \
        bash "${PROFILE_SCRIPT}" collect
      ;;
  esac
}

data_jobs=(fixed-2 profile-qwenoft profile-qwenpi_v3 profile-qwengr00t)
data_gpus=(6 7)
declare -a data_pids=("" "")
declare -a data_names=("" "")
data_next=0
data_done=0

while (( data_done < ${#data_jobs[@]} )); do
  for index in "${!data_gpus[@]}"; do
    pid="${data_pids[${index}]}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then
        record data-complete "${data_names[${index}]}" "gpu=${data_gpus[${index}]}"
        data_done=$((data_done + 1))
      else
        record data-failed "${data_names[${index}]}" "gpu=${data_gpus[${index}]}"
        touch "${QUEUE_ROOT}/data.failed"
        exit 1
      fi
      data_pids[${index}]=""
      data_names[${index}]=""
    fi
    if [[ -z "${data_pids[${index}]}" ]] && (( data_next < ${#data_jobs[@]} )); then
      job="${data_jobs[${data_next}]}"
      gpu="${data_gpus[${index}]}"
      record data-start "${job}" "gpu=${gpu}"
      run_data_job "${job}" "${gpu}" > "${QUEUE_ROOT}/data-${job}.log" 2>&1 &
      data_pids[${index}]=$!
      data_names[${index}]="${job}"
      data_next=$((data_next + 1))
    fi
  done
  sleep 10
done
touch "${QUEUE_ROOT}/data.complete"

run_training_job() {
  local condition="$1"
  local model="$2"
  local devices="$3"
  local port="$4"
  if [[ "${condition}" == profile ]]; then
    PROFILE_MODELS="${model}" PROFILE_TRAIN_DEVICES="${devices}" \
      PROFILE_TRAIN_PORTS="${port}" bash "${PROFILE_SCRIPT}" train-eval
  else
    FIXED_MODELS="${model}" FIXED_TRAIN_DEVICES="${devices}" \
      FIXED_TRAIN_PORTS="${port}" bash "${FIXED_SCRIPT}" train-eval
  fi
}

training_jobs=(
  fixed-2:qwengr00t
  profile:qwengr00t
  fixed-2:qwenpi_v3
  profile:qwenpi_v3
  fixed-2:qwenoft
  profile:qwenoft
)
slot_devices=(0,1 2,3 4,5 6,7)
slot_ports=(29640 29642 29644 29646)
slot_markers=(
  "${H1_ROOT}/mikasa_qwenoft_latency_3090_2xh200_20260922/verification.json"
  "${H1_ROOT}/mikasa_qwenpi_v3_latency_3090_2xh200_20260922/verification.json"
  "${H1_ROOT}/mikasa_qwengr00t_latency_3090_2xh200_20260922/verification.json"
  ""
)
declare -a slot_pids=("" "" "" "")
declare -a slot_jobs=("" "" "" "")
training_next=0
training_done=0
training_failed=0

while (( training_done < ${#training_jobs[@]} )); do
  for index in "${!slot_devices[@]}"; do
    pid="${slot_pids[${index}]}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      if wait "${pid}"; then
        record training-complete "${slot_jobs[${index}]}" "devices=${slot_devices[${index}]}"
      else
        record training-failed "${slot_jobs[${index}]}" "devices=${slot_devices[${index}]}"
        training_failed=1
      fi
      training_done=$((training_done + 1))
      slot_pids[${index}]=""
      slot_jobs[${index}]=""
    fi
    marker="${slot_markers[${index}]}"
    if [[ -z "${slot_pids[${index}]}" ]] && \
       [[ -z "${marker}" || -f "${marker}" ]] && \
       (( training_next < ${#training_jobs[@]} )); then
      job="${training_jobs[${training_next}]}"
      condition="${job%%:*}"
      model="${job#*:}"
      devices="${slot_devices[${index}]}"
      port="${slot_ports[${index}]}"
      record training-start "${job}" "devices=${devices} port=${port}"
      run_training_job "${condition}" "${model}" "${devices}" "${port}" \
        > "${QUEUE_ROOT}/training-${condition}-${model}.log" 2>&1 &
      slot_pids[${index}]=$!
      slot_jobs[${index}]="${job}"
      training_next=$((training_next + 1))
    fi
  done
  sleep 10
done

if (( training_failed )); then
  touch "${QUEUE_ROOT}/training.failed"
  exit 1
fi

bash "${PROFILE_SCRIPT}" verify > "${OUTPUT_ROOT}/profile-verification.json"
bash "${FIXED_SCRIPT}" verify > "${OUTPUT_ROOT}/fixed-verification.json"
touch "${QUEUE_ROOT}/complete"
