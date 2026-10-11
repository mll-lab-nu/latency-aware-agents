#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <checkpoint-root> <output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHON="${PYTHON:-python}"

MAX_EPISODES=100 MAX_STEPS=3600 VAL_FRACTION=0.1 SPLIT_SEED=0 \
NUM_ENVS="${NUM_ENVS:-12}" REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS:-16}" \
DEVICE="${DEVICE:-gpu}" DETERMINISTIC=false EXPORT_IMAGE_SIZE=224,224 \
IMAGE_WRITER_WORKERS="${IMAGE_WRITER_WORKERS:-16}" FILTER_PRESET=none \
EPISODE_RETURN_GT=2000 CONTEXT_WINDOW=0 \
GYM_MAKE_KWARGS_JSON='{"noop_max":30,"fire_reset":true}' \
bash "$REPO_ROOT/scripts/bash_scripts/run_sf_teacher_dataset_export.sh" "$1" "$2"
