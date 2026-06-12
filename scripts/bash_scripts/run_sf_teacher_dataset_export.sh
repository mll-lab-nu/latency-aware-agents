#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat >&2 <<'EOF'
Usage:
  bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh <checkpoint-root> <output-dir>

Example:
  bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
    results/checkpoints/flappy_sf_frame_stack_4_seed0 \
    /tmp/flappy_teacher_export_smoke

Optional environment variables:
  MAX_EPISODES        Number of episodes to collect. Default: 1k.
  MAX_STEPS           Max steps per episode. Default: 1k.
  VAL_FRACTION        Validation split fraction. Default: 0.01.
  NUM_ENVS            Number of rollout envs. Default: 12.
  DEVICE              Sample Factory device override. Default: cpu.
  LATENCY_CONFIG      YAML file with a training-compatible latency: section.
  EXPORT_IMAGE_SIZE   Optional saved image size, e.g. 512,288.
  IMAGE_WRITER_WORKERS Optional async PNG writer threads. Default: exporter default.
  HF_REPO_ID          If set, upload the export to this Hugging Face dataset repo.
  HF_PRIVATE          Set to 1 to upload as private. Default: 1.
  HF_TOKEN            Hugging Face token used when HF_REPO_ID is set.

Notes:
  By default the exporter uses the latency config saved in the checkpoint config.json.
  The exported images are env raw RGB frames.
  Dataset row prompts are fixed in latency_bench/data/prompts.py.
EOF
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

CHECKPOINT_ROOT="$1"
OUTPUT_DIR="$2"

MAX_EPISODES="${MAX_EPISODES:-1000}"
MAX_STEPS="${MAX_STEPS:-1000}"
VAL_FRACTION="${VAL_FRACTION:-0.01}"
NUM_ENVS="${NUM_ENVS:-12}"
DEVICE="${DEVICE:-cpu}"
HF_PRIVATE="${HF_PRIVATE:-0}"
IMAGE_WRITER_WORKERS="${IMAGE_WRITER_WORKERS:-16}"

CMD=(
    python scripts/rollout_data/export_sf_teacher_dataset.py
    --checkpoint-root "${CHECKPOINT_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --max-episodes "${MAX_EPISODES}"
    --max-steps-per-episode "${MAX_STEPS}"
    --val-fraction "${VAL_FRACTION}"
    --num-envs "${NUM_ENVS}"
    --device "${DEVICE}"
)

if [[ -n "${LATENCY_CONFIG:-}" ]]; then
    CMD+=(--latency-config "${LATENCY_CONFIG}")
fi

if [[ -n "${EXPORT_IMAGE_SIZE:-}" ]]; then
    CMD+=(--export-image-size "${EXPORT_IMAGE_SIZE}")
fi

if [[ -n "${IMAGE_WRITER_WORKERS:-}" ]]; then
    CMD+=(--image-writer-workers "${IMAGE_WRITER_WORKERS}")
fi

if [[ -n "${HF_REPO_ID:-}" ]]; then
    CMD+=(--push-to-hub --hf-repo-id "${HF_REPO_ID}" --hf-token "${HF_TOKEN:-}")
    if [[ "${HF_PRIVATE}" == "1" || "${HF_PRIVATE}" == "true" ]]; then
        CMD+=(--hf-private)
    fi
fi

printf '==> Running teacher dataset export\n'
printf '    checkpoint: %s\n' "${CHECKPOINT_ROOT}"
printf '    output:     %s\n' "${OUTPUT_DIR}"
if [[ -n "${LATENCY_CONFIG:-}" ]]; then
    printf '    latency:    %s\n' "${LATENCY_CONFIG}"
else
    printf '    latency:    checkpoint config.json\n'
fi
if [[ -n "${HF_REPO_ID:-}" ]]; then
    printf '    upload:     %s\n' "${HF_REPO_ID}"
fi
if [[ -n "${IMAGE_WRITER_WORKERS:-}" ]]; then
    printf '    image writer workers: %s\n' "${IMAGE_WRITER_WORKERS}"
fi

"${CMD[@]}"
