"""Package successful native MIKASA H1 NPZ episodes directly as LeRobot data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latency_bench.integrations.starvla.task_contracts import read_task_contract, task_spec, write_task_contract
from latency_bench.integrations.starvla.task_contracts import _rewrite_dataset, _write_manifest


def _write_split(raw_root: Path, output: Path, episode_indices: list[int]) -> dict[str, int]:
    # LeRobot is an optional packaging runtime, separate from the simulator.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    spec = task_spec("mikasa")

    dataset = LeRobotDataset.create(
        repo_id=f"local/{output.parent.parent.name}-{output.name}",
        root=output,
        fps=20,
        robot_type=spec.robot_type,
        features={
            "observation.images.top": {
                "dtype": "video", "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": "video", "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {"dtype": "float32", "shape": (spec.state_dim,)},
            "action": {"dtype": "float32", "shape": (spec.action_dim,)},
        },
        use_videos=True,
    )
    frames = 0
    for episode_index in episode_indices:
        with np.load(raw_root / f"train_data_{episode_index}.npz") as episode:
            instruction = episode["language_instruction"].item()
            for frame in range(len(episode["action"])):
                dataset.add_frame({
                    "observation.images.top": episode["rgb"][frame, ..., :3],
                    "observation.images.wrist": episode["rgb"][frame, ..., 3:],
                    "observation.state": episode["proprio"][frame],
                    "action": episode["action"][frame],
                    "task": instruction,
                })
                frames += 1
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    modality = {
        "state": {"proprio": {
            "start": 0, "end": spec.state_dim, "original_key": "observation.state",
        }},
        "action": {"target": {
            "start": 0, "end": spec.action_dim, "original_key": spec.action_key,
        }},
        "video": {
            key.removeprefix("video."): {
                "original_key": f"observation.images.{key.removeprefix('video.')}"
            }
            for key in spec.video_keys
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (output / "meta/modality.json").write_text(json.dumps(modality, indent=2) + "\n")
    return {"episodes": len(episode_indices), "frames": frames}


def prepare(raw_root: Path, output_root: Path, *, train_episodes: int, val_episodes: int) -> dict:
    """Package successful H1 splits and share training normalization with validation."""
    metadata = json.loads((raw_root / "metadata.json").read_text())
    successful_indices = [index for index, success in enumerate(metadata["success_once"]) if success]
    train_indices = successful_indices[:train_episodes]
    val_indices = successful_indices[train_episodes:train_episodes + val_episodes]
    train = output_root / "lerobot"
    val = output_root / "lerobot__val"
    counts = {
        "train": _write_split(raw_root, train, train_indices),
        "validation": _write_split(raw_root, val, val_indices),
    }
    spec = task_spec("mikasa")
    for dataset_dir in (train, val):
        _rewrite_dataset(dataset_dir, spec)
    write_task_contract(train, spec=spec)
    write_task_contract(val, spec=spec, normalization_source=train)
    for dataset_dir in (train, val):
        _write_manifest(dataset_dir, spec, read_task_contract(dataset_dir))
    report = {
        "conversion": "direct_npz_to_lerobot",
        "raw_root": str(raw_root), "fps": spec.fps, "counts": counts,
        "train_episode_indices": train_indices, "validation_episode_indices": val_indices,
    }
    (output_root / "packaging.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=250)
    parser.add_argument("--val-episodes", type=int, default=25)
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.raw_root, args.output_root,
        train_episodes=args.train_episodes, val_episodes=args.val_episodes,
    ), indent=2))


if __name__ == "__main__":
    main()
