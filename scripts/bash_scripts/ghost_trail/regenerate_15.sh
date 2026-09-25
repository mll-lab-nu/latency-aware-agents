#!/usr/bin/env bash
set -euo pipefail

# Regenerate the same accepted Demon Attack episodes as the 30-frame ghost-trail
# dataset, but render observations with a 15-frame continuous ghost trail.
#
# On a new machine, edit only this variable if your workspace root differs.
WORKSPACE="${WORKSPACE:-/workspace}"

SOURCE_GHOST_HF_REPO="placeholder/demon_attack_ghost30_200ep"
CHECKPOINT_HF_REPO="placeholder/latency-checkpoints"
CHECKPOINT_HF_INCLUDE="checkpoints_old/demon_attack_sf_same_latency_l6_seed0/**"
CHECKPOINT_LOCAL_DIR="${WORKSPACE}/small_model"
CHECKPOINT_ROOT="${WORKSPACE}/small_model/checkpoints_old/demon_attack_sf_same_latency_l6_seed0"
OUTPUT_DIR="${WORKSPACE}/outputs/teacher_datasets/demon_attack_ghost15_same_episodes_200ep"
UPLOAD_HF_REPO="placeholder/demon_attack_ghost15_200ep"
UPLOAD_HF_PATH="demon_latency_6"

MAX_EPISODES=200
MAX_STEPS=3600
VAL_FRACTION=0.1
FILTER_PRESET=clean_v1
DEVICE=cuda
NUM_ENVS=2048
REPLAY_NUM_ENVS=16
IMAGE_WRITER_WORKERS=32
GHOST_TRAIL_HISTORY_FRAMES=15

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

echo "==> Recovering accepted episode seeds from ${SOURCE_GHOST_HF_REPO}"
OUTPUT_DIR="${OUTPUT_DIR}" \
SOURCE_GHOST_HF_REPO="${SOURCE_GHOST_HF_REPO}" \
python - <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset

repo = os.environ["SOURCE_GHOST_HF_REPO"]
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset(repo)
specs = {}
for split_name, split in ds.items():
    rows = split.select_columns(["episode_idx", "seed", "split", "decision_step"])
    for row in rows:
        episode_idx = int(row["episode_idx"])
        spec = specs.setdefault(
            episode_idx,
            {
                "attempt_idx": episode_idx,
                "episode_idx": episode_idx,
                "split": str(row.get("split") or split_name),
                "seed": int(row["seed"]),
                "episode_length": 0,
                "episode_raw_return": 0.0,
            },
        )
        spec["episode_length"] = max(spec["episode_length"], int(row["decision_step"]) + 1)

runtime_metadata = {
    "checkpoint_train_step": 0,
    "frame_stack": 4,
    "env_fps": 60.0,
    "obs_fps": 15.0,
}
try:
    from huggingface_hub import hf_hub_download

    metadata_path = hf_hub_download(repo_id=repo, repo_type="dataset", filename="metadata.json")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    for key in runtime_metadata:
        if key in metadata:
            runtime_metadata[key] = metadata[key]
except Exception as exc:
    print(f"[ghost15-replay] warning: could not load metadata.json from {repo}: {exc}")

probe_state = {
    "next_attempt_idx": len(specs),
    "attempted_episodes": len(specs),
    "rejected_episodes": 0,
    "accepted_specs": [specs[idx] for idx in sorted(specs)],
    "runtime_metadata": runtime_metadata,
}

probe_state_path = output_dir / "probe_state.json"
probe_state_path.write_text(json.dumps(probe_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[ghost15-replay] wrote {len(specs)} accepted episode specs to {probe_state_path}")
PY

echo "==> Replaying same episodes with ${GHOST_TRAIL_HISTORY_FRAMES}-frame ghost trail"
IMAGE_TRANSFORM=demon_attack_ghost_trail \
GHOST_TRAIL_HISTORY_FRAMES="${GHOST_TRAIL_HISTORY_FRAMES}" \
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

echo "==> Wrote ghost15 dataset to ${OUTPUT_DIR}"

echo "==> Uploading ghost15 dataset to ${UPLOAD_HF_REPO}/${UPLOAD_HF_PATH}"
hf upload "${UPLOAD_HF_REPO}" \
    "${OUTPUT_DIR}" \
    "${UPLOAD_HF_PATH}" \
    --repo-type dataset \
    --exclude "sample_factory_init.log" \
    --commit-message "Upload demon attack ghost trail 15 teacher dataset"
