#!/usr/bin/env bash

set -euo pipefail

for arg in "$@"; do
  case "${arg}" in
    --latency-distribution|--latency-distribution=*|\
    --latency-mean-frames|--latency-mean-frames=*|\
    --latency-std-frames|--latency-std-frames=*|\
    --latency-min-frames|--latency-min-frames=*|\
    --latency-max-frames|--latency-max-frames=*)
      echo "Do not pass latency distribution flags to gen_config_norm_and_uni.sh; edit the case tables in this script instead." >&2
      exit 2
      ;;
  esac
done

EXTRA_ARGS=("$@")
IFS=',' read -r -a ENVS_LIST <<< "${ENVS:-flappy,demon_attack,deadly_corridor}"
SEEDS="${SEEDS:-0}"
GENERATED_ROOT="${GENERATED_ROOT:-configs/train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
DEVICE="${DEVICE:-gpu}"

if [[ "${#ENVS_LIST[@]}" -eq 0 ]]; then
  echo "ENVS must contain at least one environment." >&2
  exit 2
fi

generate_for_envs() {
  local family="$1"
  shift

  local env
  for env in "${ENVS_LIST[@]}"; do
    if [[ -z "${env}" ]]; then
      continue
    fi

    ENVS="${env}" \
    SEEDS="${SEEDS}" \
    GENERATED_ROOT="${GENERATED_ROOT}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DEVICE="${DEVICE}" \
    bash scripts/bash_scripts/generate_config.sh \
      --wandb-group "${env//_/-}-${family}" \
      "$@" \
      "${EXTRA_ARGS[@]}"
  done
}

generate_normal_case() {
  local name="$1"
  local mean="$2"
  local std="$3"
  local min="$4"
  local max="$5"

  echo "[normal] ${name}: mean=${mean}, std=${std}, min=${min}, max=${max}"
  generate_for_envs normal \
    --latency-distribution normal \
    --latency-mean-frames "${mean}" \
    --latency-std-frames "${std}" \
    --latency-min-frames "${min}" \
    --latency-max-frames "${max}"
}

generate_uniform_case() {
  local name="$1"
  local min="$2"
  local max="$3"

  echo "[uniform] ${name}: min=${min}, max=${max}"
  generate_for_envs uniform \
    --latency-distribution uniform \
    --latency-min-frames "${min}" \
    --latency-max-frames "${max}"
}

NORMAL_CASES=(
  "near_zero 1 0.5 0 15"
  "low_stable 2 0.5 0 15"
  "low_jitter 2 2 0 15"
  "medium_stable 4 0.5 0 15"
  "medium_jitter 4 2 0 15"
  "medium_high_jitter 4 4 0 15"
  "high_stable 8 0.5 0 15"
  "high_jitter 8 2.5 0 15"
)

UNIFORM_CASES=(
  "low_zero 0 2"
  "low_no_zero 1 3"
  "low_medium 0 5"
  "medium 3 7"
  "medium_high_jitter 0 10"
  "high_medium_jitter 6 10"
  "high_high_jitter 4 12"
  "full_range 0 15"
)

for row in "${NORMAL_CASES[@]}"; do
  read -r name mean std min max <<< "${row}"
  generate_normal_case "${name}" "${mean}" "${std}" "${min}" "${max}"
done

for row in "${UNIFORM_CASES[@]}"; do
  read -r name min max <<< "${row}"
  generate_uniform_case "${name}" "${min}" "${max}"
done
