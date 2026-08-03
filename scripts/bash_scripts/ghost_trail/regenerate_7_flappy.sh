#!/usr/bin/env bash
set -euo pipefail

# Regenerate the same accepted Flappy Bird episodes as the memory rollout
# dataset below, but render observations with a 15-frame Flappy ghost trail.
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# On a new machine, override WORKSPACE if you want downloads/outputs elsewhere.
WORKSPACE="${WORKSPACE:-${REPO_ROOT}}"

SOURCE_HF_REPO="latency-sensitive-bench/memory-rollouts"
SOURCE_HF_CONFIG="flappy_fixed_latency_3_200ep_7k2steps"
SOURCE_HF_REVISION="6875117373bb24255a07653b65501b53a35dc2d4"

CHECKPOINT_HF_REPO="latency-sensitive-bench/paper-experiment-models"
CHECKPOINT_HF_INCLUDE="memory/rollout_teachers/flappy/train/memory_rollout_teacher:flappy:teacher:fixed_l3:fs4:obs30:stride1:seed0/**"
CHECKPOINT_LOCAL_DIR="${WORKSPACE}/paper_experiment_models"
CHECKPOINT_ROOT="${CHECKPOINT_LOCAL_DIR}/memory/rollout_teachers/flappy/train/memory_rollout_teacher:flappy:teacher:fixed_l3:fs4:obs30:stride1:seed0"
OUTPUT_DIR="${WORKSPACE}/outputs/teacher_datasets/flappy_fixed_latency_3_200ep_7k2steps_ghost7"
UPLOAD_HF_REPO="latency-sensitive-bench/memory-rollouts"
UPLOAD_HF_PATH="flappy_fixed_latency_3_200ep_7k2steps_ghost7"

MAX_EPISODES=200
MAX_STEPS=7200
VAL_FRACTION=0.1
# Do not apply any new episode-level gate or within-episode row filtering while
# replaying the source dataset's accepted seeds.
FILTER_PRESET=none
DEVICE=cuda
NUM_ENVS=2048
REPLAY_NUM_ENVS=16
IMAGE_WRITER_WORKERS=32
GHOST_TRAIL_HISTORY_FRAMES=7
GHOST_TRAIL_GAMMA=1.3
GHOST_TRAIL_MIN_ALPHA=35
GHOST_TRAIL_SCROLL_PX_PER_STEP=4.0
GHOST_TRAIL_GROUND_FRACTION=0.22

cd "${REPO_ROOT}"

if ! command -v python >/dev/null 2>&1; then
    echo "error: python not found" >&2
    exit 127
fi
if ! command -v hf >/dev/null 2>&1; then
    echo "error: hf not found" >&2
    exit 127
fi

echo "==> Downloading checkpoint into ${CHECKPOINT_LOCAL_DIR}"
hf download "${CHECKPOINT_HF_REPO}" \
    --repo-type model \
    --include "${CHECKPOINT_HF_INCLUDE}" \
    --local-dir "${CHECKPOINT_LOCAL_DIR}"

if [[ ! -e "${CHECKPOINT_ROOT}" ]]; then
    echo "error: checkpoint root does not exist after download: ${CHECKPOINT_ROOT}" >&2
    exit 1
fi

echo "==> Recovering accepted episode seeds from ${SOURCE_HF_REPO}/${SOURCE_HF_CONFIG}"
OUTPUT_DIR="${OUTPUT_DIR}" \
SOURCE_HF_REPO="${SOURCE_HF_REPO}" \
SOURCE_HF_CONFIG="${SOURCE_HF_CONFIG}" \
SOURCE_HF_REVISION="${SOURCE_HF_REVISION}" \
python - <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset


repo = os.environ["SOURCE_HF_REPO"]
config = os.environ["SOURCE_HF_CONFIG"]
revision = os.environ["SOURCE_HF_REVISION"]
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset(repo, config, revision=revision, streaming=True)
specs = {}
for split_name, split in ds.items():
    rows = split.select_columns(["episode_idx", "seed", "split", "decision_step"])
    for row in rows:
        episode_idx = int(row["episode_idx"])
        row_split = str(row.get("split") or split_name)
        if row_split == "validation":
            row_split = "val"
        spec = specs.setdefault(
            episode_idx,
            {
                "attempt_idx": episode_idx,
                "episode_idx": episode_idx,
                "split": row_split,
                "seed": int(row["seed"]),
                "episode_length": 0,
                "episode_raw_return": 0.0,
            },
        )
        spec["episode_length"] = max(spec["episode_length"], int(row["decision_step"]) + 1)

probe_state = {
    "next_attempt_idx": len(specs),
    "attempted_episodes": len(specs),
    "rejected_episodes": 0,
    "accepted_specs": [specs[idx] for idx in sorted(specs)],
    "runtime_metadata": {
        "checkpoint_train_step": 0,
        "frame_stack": 4,
        "env_fps": 30.0,
        "obs_fps": 30.0,
    },
}

probe_state_path = output_dir / "probe_state.json"
probe_state_path.write_text(json.dumps(probe_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[flappy-ghost15-replay] wrote {len(specs)} accepted episode specs to {probe_state_path}")
PY

echo "==> Replaying same episodes with ${GHOST_TRAIL_HISTORY_FRAMES}-frame Flappy ghost trail"
IMAGE_TRANSFORM=flappy_ghost_trail \
GHOST_TRAIL_HISTORY_FRAMES="${GHOST_TRAIL_HISTORY_FRAMES}" \
GHOST_TRAIL_GAMMA="${GHOST_TRAIL_GAMMA}" \
GHOST_TRAIL_MIN_ALPHA="${GHOST_TRAIL_MIN_ALPHA}" \
GHOST_TRAIL_SCROLL_PX_PER_STEP="${GHOST_TRAIL_SCROLL_PX_PER_STEP}" \
GHOST_TRAIL_GROUND_FRACTION="${GHOST_TRAIL_GROUND_FRACTION}" \
MAX_EPISODES="${MAX_EPISODES}" \
MAX_STEPS="${MAX_STEPS}" \
VAL_FRACTION="${VAL_FRACTION}" \
FILTER_PRESET="${FILTER_PRESET}" \
DEVICE="${DEVICE}" \
NUM_ENVS="${NUM_ENVS}" \
REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS}" \
IMAGE_WRITER_WORKERS="${IMAGE_WRITER_WORKERS}" \
bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
    "${CHECKPOINT_ROOT}" \
    "${OUTPUT_DIR}"

echo "==> Wrote Flappy ghost 7 dataset to ${OUTPUT_DIR}"

echo "==> Uploading Flappy ghost 7 dataset to ${UPLOAD_HF_REPO}/${UPLOAD_HF_PATH}"
hf upload "${UPLOAD_HF_REPO}" \
    "${OUTPUT_DIR}" \
    "${UPLOAD_HF_PATH}" \
    --repo-type dataset \
    --exclude "sample_factory_init.log" \
    --commit-message "Upload flappy fixed latency 3 ghost trail 7 memory rollout dataset"
