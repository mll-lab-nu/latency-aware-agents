#!/usr/bin/env python3
"""Package conditioned MIKASA H8 rollouts as LeRobot v3 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latency_bench.data.starvla_tasks import read_task_contract, task_spec, write_task_contract
from scripts.starvla.prepare_dataset import _rewrite_dataset, _write_manifest


INSTRUCTION = "Intercept the rolling ball and grasp it to stop it."


def _features() -> dict:
    return {
        "observation.images.top": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (128, 128, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (71,)},
        "action": {"dtype": "float32", "shape": (7,)},
        "issued_command_chunk": {"dtype": "float32", "shape": (8, 7)},
        "action_prefix_mask": {"dtype": "bool", "shape": (8,)},
        "action_loss_mask": {"dtype": "bool", "shape": (8,)},
    }


def _write_split(raw: Path, output: Path, episode_indices: list[int]) -> dict[str, int]:
    # LeRobot is an optional packaging runtime; collection and unit tests do not import it.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"local/{output.parent.name}-{output.name}",
        root=output,
        fps=20,
        robot_type="latency_mikasa_conditioned_h8",
        features=_features(),
        use_videos=True,
    )
    frames = 0
    for episode_index in episode_indices:
        episode = np.load(raw / f"episode_{episode_index:06d}.npz")
        for frame in range(len(episode["state"])):
            dataset.add_frame(
                {
                    "observation.images.top": episode["top"][frame],
                    "observation.images.wrist": episode["wrist"][frame],
                    "observation.state": episode["state"][frame],
                    "action": episode["action"][frame],
                    "issued_command_chunk": episode["issued_command_chunk"][frame],
                    "action_prefix_mask": episode["action_prefix_mask"][frame],
                    "action_loss_mask": episode["action_loss_mask"][frame],
                    "task": INSTRUCTION,
                }
            )
            frames += 1
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    (output / "meta/modality.json").write_text(
        json.dumps(
            {
                "state": {
                    "proprio": {
                        "start": 0,
                        "end": 71,
                        "original_key": "observation.state",
                    }
                },
                "action": {
                    "target": {
                        "start": 0,
                        "end": 7,
                        "original_key": "issued_command_chunk",
                    }
                },
                "video": {
                    "top": {"original_key": "observation.images.top"},
                    "wrist": {"original_key": "observation.images.wrist"},
                },
                "annotation": {
                    "human.task_description": {"original_key": "task_index"}
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"episodes": len(episode_indices), "frames": frames}


def prepare(
    raw_root: Path,
    output_train: Path,
) -> dict:
    collection = json.loads((raw_root / "collection.json").read_text(encoding="utf-8"))
    output_val = output_train.with_name(f"{output_train.name}__val")
    counts = {
        "train": _write_split(
            raw_root / "episodes",
            output_train,
            collection["splits"]["train"]["episode_indices"],
        ),
        "validation": _write_split(
            raw_root / "episodes",
            output_val,
            collection["splits"]["validation"]["episode_indices"],
        ),
    }
    spec = task_spec("mikasa_conditioned_h8")
    for dataset_dir in (output_train, output_val):
        _rewrite_dataset(dataset_dir, spec)
    write_task_contract(output_train, spec=spec)
    write_task_contract(output_val, spec=spec, normalization_source=output_train)
    label_provenance = {
        "semantics": "zero_h1_teacher_after_causal_prefix",
        "teacher_checkpoint": collection["teacher_checkpoint"],
        "condition": collection["condition"],
        "model": collection["model"],
        "latency": collection["latency_config"]["latency"],
        "action_prefix": collection["action_prefix"],
    }
    for dataset_dir in (output_train, output_val):
        contract_path = dataset_dir / "task_contract.json"
        contract = read_task_contract(dataset_dir)
        contract["action_prefix"] = collection["action_prefix"]
        contract["label_provenance"] = label_provenance
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        info_path = dataset_dir / "meta/info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["label_provenance"] = label_provenance
        info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        _write_manifest(dataset_dir, spec, contract)
    result = {
        "format": "latency_bench_starvla_h8_v2",
        "task": spec.name,
        "robot_type": spec.robot_type,
        "raw_collection": str(raw_root),
        "action_prefix": collection["action_prefix"],
        "splits": counts,
        "train": str(output_train),
        "validation": str(output_val),
    }
    (output_train.parent / "provenance.json").write_text(
        json.dumps({**collection, "packaged": result}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_train.parent / "README.md").write_text(
        "# MIKASA InterceptGrabFast conditioned H8 teacher data\n\n"
        "Conditioned causal-prefix rollouts with zero-latency H1 teacher suffix labels. "
        "Train and validation are split by disjoint episode seeds and retain both successful "
        "and failed main-environment attempts with valid teacher targets.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw_root, args.output_train), indent=2))


if __name__ == "__main__":
    main()
