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
  MAX_STEPS           Max policy decision steps per episode. Default: 1k.
  VAL_FRACTION        Validation split fraction. Default: 0.01.
  NUM_ENVS            Number of probe rollout envs. Default: 12.
  REPLAY_NUM_ENVS     Number of replay rollout envs. Default: 16.
  DEVICE              Sample Factory device override. Default: cpu.
  LATENCY_CONFIG      YAML file with a training-compatible latency: section.
  EXPORT_IMAGE_SIZE   Optional saved image size, e.g. 512,288.
  IMAGE_TRANSFORM     Observation image transform: raw_rgb, flappy_ghost_trail, or demon_attack_ghost_trail. Default: raw_rgb.
  GHOST_TRAIL_HISTORY_FRAMES  flappy_ghost_trail: past decision-step frames (default 5). demon_attack_ghost_trail: total consecutive raw frames in the trail window (use 30 for the validated continuous_30 ablation).
  GHOST_TRAIL_GAMMA   For flappy_ghost_trail, alpha decay gamma. Default: 1.3.
  GHOST_TRAIL_MIN_ALPHA  For flappy_ghost_trail, oldest ghost opacity floor. Default: 35.
  GHOST_TRAIL_SCROLL_PX_PER_STEP  For flappy_ghost_trail, Flappy world-scroll pixels per decision step. Default: 4.0.
  GHOST_TRAIL_GROUND_FRACTION  For flappy_ghost_trail, bottom-frame occlusion band. Default: 0.22.
  IMAGE_WRITER_WORKERS Optional async PNG writer threads. Default: exporter default.
  HF_REPO_ID          If set, upload the export to this Hugging Face dataset repo.
  HF_PRIVATE          Set to 1 to upload as private. Default: 1.
  HF_TOKEN            Hugging Face token used when HF_REPO_ID is set.
  HF_CONFIG_NAME      HF dataset config name for multi-config repos.
  FILTER_PRESET       Filter preset: none or clean_v1. Default: clean_v1.
  DEMON_EPISODE_THRESHOLD  Override Demon Attack episode clipped return threshold.

Notes:
  By default the exporter parses *_same_latency_lN_* checkpoint names when present,
  otherwise it uses the latency config saved in checkpoint config.json.
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
REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS:-16}"
DEVICE="${DEVICE:-cpu}"
HF_PRIVATE="${HF_PRIVATE:-0}"
IMAGE_WRITER_WORKERS="${IMAGE_WRITER_WORKERS:-16}"
IMAGE_TRANSFORM="${IMAGE_TRANSFORM:-raw_rgb}"
GHOST_TRAIL_HISTORY_FRAMES="${GHOST_TRAIL_HISTORY_FRAMES:-5}"
GHOST_TRAIL_GAMMA="${GHOST_TRAIL_GAMMA:-1.3}"
GHOST_TRAIL_MIN_ALPHA="${GHOST_TRAIL_MIN_ALPHA:-35}"
GHOST_TRAIL_SCROLL_PX_PER_STEP="${GHOST_TRAIL_SCROLL_PX_PER_STEP:-4.0}"
GHOST_TRAIL_GROUND_FRACTION="${GHOST_TRAIL_GROUND_FRACTION:-0.22}"

CMD=(
    python scripts/rollout_data/export_sf_teacher_dataset.py
    --checkpoint-root "${CHECKPOINT_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --max-episodes "${MAX_EPISODES}"
    --max-steps-per-episode "${MAX_STEPS}"
    --val-fraction "${VAL_FRACTION}"
    --num-envs "${NUM_ENVS}"
    --replay-num-envs "${REPLAY_NUM_ENVS}"
    --device "${DEVICE}"
    --image-transform "${IMAGE_TRANSFORM}"
)

if [[ "${IMAGE_TRANSFORM}" == "flappy_ghost_trail" || "${IMAGE_TRANSFORM}" == "demon_attack_ghost_trail" ]]; then
    CMD+=(
        --ghost-trail-history-frames "${GHOST_TRAIL_HISTORY_FRAMES}"
        --ghost-trail-gamma "${GHOST_TRAIL_GAMMA}"
        --ghost-trail-min-alpha "${GHOST_TRAIL_MIN_ALPHA}"
        --ghost-trail-scroll-px-per-step "${GHOST_TRAIL_SCROLL_PX_PER_STEP}"
        --ghost-trail-ground-fraction "${GHOST_TRAIL_GROUND_FRACTION}"
    )
fi

if [[ -n "${LATENCY_CONFIG:-}" ]]; then
    CMD+=(--latency-config "${LATENCY_CONFIG}")
fi

if [[ -n "${EXPORT_IMAGE_SIZE:-}" ]]; then
    CMD+=(--export-image-size "${EXPORT_IMAGE_SIZE}")
fi

if [[ -n "${IMAGE_WRITER_WORKERS:-}" ]]; then
    CMD+=(--image-writer-workers "${IMAGE_WRITER_WORKERS}")
fi

if [[ -n "${FILTER_PRESET:-}" ]]; then
    CMD+=(--filter-preset "${FILTER_PRESET}")
fi

if [[ -n "${DEMON_EPISODE_THRESHOLD:-}" ]]; then
    CMD+=(--demon-episode-threshold "${DEMON_EPISODE_THRESHOLD}")
fi

if [[ -n "${HF_REPO_ID:-}" ]]; then
    CMD+=(--push-to-hub --hf-repo-id "${HF_REPO_ID}" --hf-token "${HF_TOKEN:-}")
    if [[ "${HF_PRIVATE}" == "1" || "${HF_PRIVATE}" == "true" ]]; then
        CMD+=(--hf-private)
    fi
    if [[ -n "${HF_CONFIG_NAME:-}" ]]; then
        CMD+=(--hf-config-name "${HF_CONFIG_NAME}")
    fi
fi

printf '==> Running teacher dataset export\n'
printf '    checkpoint: %s\n' "${CHECKPOINT_ROOT}"
printf '    output:     %s\n' "${OUTPUT_DIR}"
printf '    probe envs: %s\n' "${NUM_ENVS}"
printf '    replay envs: %s\n' "${REPLAY_NUM_ENVS}"
printf '    image transform: %s\n' "${IMAGE_TRANSFORM}"
if [[ "${IMAGE_TRANSFORM}" == "flappy_ghost_trail" || "${IMAGE_TRANSFORM}" == "demon_attack_ghost_trail" ]]; then
    printf '    ghost history frames: %s\n' "${GHOST_TRAIL_HISTORY_FRAMES}"
    printf '    ghost gamma: %s\n' "${GHOST_TRAIL_GAMMA}"
    printf '    ghost min alpha: %s\n' "${GHOST_TRAIL_MIN_ALPHA}"
    printf '    ghost scroll px/step: %s\n' "${GHOST_TRAIL_SCROLL_PX_PER_STEP}"
    printf '    ghost ground fraction: %s\n' "${GHOST_TRAIL_GROUND_FRACTION}"
fi
if [[ -n "${LATENCY_CONFIG:-}" ]]; then
    printf '    latency:    %s\n' "${LATENCY_CONFIG}"
else
    printf '    latency:    checkpoint name if *_same_latency_lN_* else config.json\n'
fi
if [[ -n "${HF_REPO_ID:-}" ]]; then
    printf '    upload:     %s\n' "${HF_REPO_ID}"
fi
if [[ -n "${IMAGE_WRITER_WORKERS:-}" ]]; then
    printf '    image writer workers: %s\n' "${IMAGE_WRITER_WORKERS}"
fi
printf '    filter preset: %s\n' "${FILTER_PRESET:-clean_v1}"
if [[ -n "${DEMON_EPISODE_THRESHOLD:-}" ]]; then
    printf '    demon threshold: %s\n' "${DEMON_EPISODE_THRESHOLD}"
fi
if [[ -n "${HF_CONFIG_NAME:-}" ]]; then
    printf '    hf config name: %s\n' "${HF_CONFIG_NAME}"
fi

"${CMD[@]}"
