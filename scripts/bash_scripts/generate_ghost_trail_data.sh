IMAGE_TRANSFORM=demon_attack_ghost_trail \
GHOST_TRAIL_HISTORY_FRAMES=30 \
MAX_EPISODES=200 MAX_STEPS=3600 VAL_FRACTION=0.1 FILTER_PRESET=clean_v1 \
DEVICE=cuda NUM_ENVS=2048 REPLAY_NUM_ENVS=16 IMAGE_WRITER_WORKERS=32 \
bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
    /workspace/small_model/checkpoints_old/demon_attack_sf_same_latency_l6_seed0 \
    /workspace/outputs/teacher_datasets/demon_attack_ghost30_200ep

# Recreate a non-ghost/raw-RGB dataset for the exact same accepted episodes.
#
# The ghost-trail images cannot be inverted back to raw frames. Instead, recover the
# accepted episode reset seeds from the already-pushed HF dataset, write a replay
# probe_state.json, and replay those seeds with IMAGE_TRANSFORM=raw_rgb.
#
# Usage:
#   REPLAY_RAW_FROM_HF=1 bash scripts/bash_scripts/generate_ghost_trail_data.sh
#
# Optional overrides:
#   GHOST_HF_REPO=MLL-Lab/LAGEN-datasets
#   RAW_REPLAY_OUTPUT_DIR=/workspace/outputs/teacher_datasets/demon_attack_raw_same_episodes_200ep
#   RAW_REPLAY_CHECKPOINT_ROOT=/workspace/small_model/checkpoints_old/demon_attack_sf_same_latency_l6_seed0
if [[ "${REPLAY_RAW_FROM_HF:-0}" == "1" ]]; then
GHOST_HF_REPO="${GHOST_HF_REPO:-MLL-Lab/LAGEN-datasets}"
GHOST_HF_SUBDIR="${GHOST_HF_SUBDIR:-experiments/visual-history/archive/demon-attack/demon_attack_ghost30_200ep/demon_latency_6}"
GHOST_HF_REVISION="${GHOST_HF_REVISION:-c65ee4c303358dc448022bed9bc25e78984449a8}"
RAW_REPLAY_OUTPUT_DIR="${RAW_REPLAY_OUTPUT_DIR:-/workspace/outputs/teacher_datasets/demon_attack_raw_same_episodes_200ep}"
RAW_REPLAY_CHECKPOINT_ROOT="${RAW_REPLAY_CHECKPOINT_ROOT:-/workspace/small_model/checkpoints_old/demon_attack_sf_same_latency_l6_seed0}"

RAW_REPLAY_OUTPUT_DIR="${RAW_REPLAY_OUTPUT_DIR}" \
GHOST_HF_REPO="${GHOST_HF_REPO}" \
GHOST_HF_SUBDIR="${GHOST_HF_SUBDIR}" \
GHOST_HF_REVISION="${GHOST_HF_REVISION}" \
python - <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset

repo = os.environ["GHOST_HF_REPO"]
output_dir = Path(os.environ["RAW_REPLAY_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

subdir = os.environ["GHOST_HF_SUBDIR"]
revision = os.environ["GHOST_HF_REVISION"]
ds = load_dataset("parquet", data_files={split: f"hf://datasets/{repo}@{revision}/{subdir}/{split}.parquet" for split in ["train", "val"]})
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

from huggingface_hub import hf_hub_download
metadata_path = hf_hub_download(
    repo_id=repo,
    repo_type="dataset",
    filename=f"{subdir}/metadata.json",
    revision=revision,
)
metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
runtime_metadata = {key: metadata[key] for key in ["checkpoint_train_step", "frame_stack", "env_fps", "obs_fps"]}

probe_state = {
    "next_attempt_idx": len(specs),
    "attempted_episodes": len(specs),
    "rejected_episodes": 0,
    "accepted_specs": [specs[idx] for idx in sorted(specs)],
    "runtime_metadata": runtime_metadata,
}

probe_state_path = output_dir / "probe_state.json"
probe_state_path.write_text(json.dumps(probe_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[raw-replay] wrote {len(specs)} accepted episode specs to {probe_state_path}")
PY

IMAGE_TRANSFORM=raw_rgb \
DEMON_EPISODE_THRESHOLD=3500 \
MAX_EPISODES=200 MAX_STEPS=3600 VAL_FRACTION=0.1 FILTER_PRESET=clean_v1 \
DEVICE=cuda NUM_ENVS=2048 REPLAY_NUM_ENVS=16 IMAGE_WRITER_WORKERS=32 \
bash scripts/bash_scripts/run_sf_teacher_dataset_export.sh \
    "${RAW_REPLAY_CHECKPOINT_ROOT}" \
    "${RAW_REPLAY_OUTPUT_DIR}"
fi