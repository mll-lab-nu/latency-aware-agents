#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <checkpoint-root> <output-dir>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"

"$PYTHON" "$REPO_ROOT/scripts/rollout_data/export_sf_teacher_dataset.py" \
  --checkpoint-root "$1" \
  --output-dir "$2" \
  --max-episodes 100 \
  --max-steps-per-episode 3600 \
  --val-fraction 0.1 \
  --split-seed 0 \
  --num-envs "${NUM_ENVS:-12}" \
  --replay-num-envs "${REPLAY_NUM_ENVS:-16}" \
  --device "${DEVICE:-gpu}" \
  --no-deterministic \
  --export-image-size 224,224 \
  --image-writer-workers "${IMAGE_WRITER_WORKERS:-16}" \
  --filter-preset none \
  --episode-return-gt 2000 \
  --context-window 0 \
  --gym-make-kwargs-json '{"noop_max":30,"fire_reset":true}'
