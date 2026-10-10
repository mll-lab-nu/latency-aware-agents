#!/usr/bin/env python3
"""Build the MIKASA H8 labels from the pinned successful H1 trajectories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from latency_bench.integrations.starvla.task_contracts import read_task_contract, task_spec, write_task_contract
from latency_bench.integrations.starvla.task_contracts import _rewrite_dataset, _write_manifest


SOURCE_REPO_ID = "MLL-Lab/LAGEN-datasets"
SOURCE_REVISION = "5421c9d272d7a939bb04045b047af84519c1fba8"
SOURCE_PATH = "zero-latency/mikasa-intercept-grab-fast/teacher-rollouts-h1-state-v1"
EXPECTED_SPLITS = {
    "train": {"episodes": 250, "anchors": 8911, "valid_action_targets": 64288},
    "validation": {"episodes": 25, "anchors": 948, "valid_action_targets": 6884},
}


def _chunk_split(dataset_dir: Path) -> dict[str, int]:
    paths = sorted((dataset_dir / "data").glob("*/*.parquet"))
    tables = [pq.read_table(path) for path in paths]
    actions = np.concatenate(
        [np.asarray(table["action"].to_pylist(), dtype=np.float32) for table in tables]
    )
    episodes = np.concatenate(
        [np.asarray(table["episode_index"].to_pylist(), dtype=np.int64) for table in tables]
    )
    chunks = np.zeros((len(actions), 8, 7), dtype=np.float32)
    masks = np.zeros((len(actions), 8), dtype=bool)
    for offset in range(min(8, len(actions))):
        available = len(actions) - offset
        same_episode = episodes[:available] == episodes[offset:]
        anchors = np.flatnonzero(same_episode)
        chunks[anchors, offset] = actions[anchors + offset]
        masks[:available, offset] = same_episode

    row_offset = 0
    for path, table in zip(paths, tables):
        row_count = len(table)
        chunk_values = pa.array(
            chunks[row_offset : row_offset + row_count].tolist(),
            type=pa.list_(pa.list_(pa.float32(), 7), 8),
        )
        mask_values = pa.array(
            masks[row_offset : row_offset + row_count].tolist(),
            type=pa.list_(pa.bool_(), 8),
        )
        pq.write_table(
            table.append_column("issued_command_chunk", chunk_values).append_column(
                "action_loss_mask", mask_values
            ),
            path,
        )
        row_offset += row_count

    info_path = dataset_dir / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["robot_type"] = "latency_mikasa_h8"
    info["features"]["issued_command_chunk"] = {
        "dtype": "float32",
        "shape": [8, 7],
        "names": None,
    }
    info["features"]["action_loss_mask"] = {
        "dtype": "bool",
        "shape": [8],
        "names": None,
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return {
        "episodes": int(np.unique(episodes).size),
        "anchors": len(actions),
        "valid_action_targets": int(masks.sum()),
    }


def prepare_mikasa_h8(
    source_train: Path | str,
    source_val: Path | str,
    output_train: Path | str,
) -> dict:
    """Copy H1 splits, add future-action labels, and write the H8 contracts."""
    source_train = Path(source_train)
    source_val = Path(source_val)
    output_train = Path(output_train)
    output_val = output_train.with_name(f"{output_train.name}__val")
    shutil.copytree(source_train, output_train)
    shutil.copytree(source_val, output_val)

    split_counts = {
        "train": _chunk_split(output_train),
        "validation": _chunk_split(output_val),
    }
    if split_counts != EXPECTED_SPLITS:
        raise AssertionError(f"Pinned MIKASA H1 split audit failed: {split_counts}")
    spec = task_spec("mikasa_h8")
    for dataset_dir in (output_train, output_val):
        _rewrite_dataset(dataset_dir, spec)
    write_task_contract(output_train, spec=spec)
    write_task_contract(output_val, spec=spec, normalization_source=output_train)
    for dataset_dir in (output_train, output_val):
        _write_manifest(dataset_dir, spec, read_task_contract(dataset_dir))

    return {
        "format": "latency_bench_starvla_h8_v1",
        "task": spec.name,
        "source_dataset": {
            "repo_id": SOURCE_REPO_ID,
            "revision": SOURCE_REVISION,
            "path": SOURCE_PATH,
        },
        "label": "observation_t -> [action_t, ..., action_t+7] within each episode",
        "splits": split_counts,
        "train": str(output_train),
        "validation": str(output_val),
    }


def prepare_mikasa_h8_from_hub(output_train: Path | str) -> dict:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            SOURCE_REPO_ID,
            repo_type="dataset",
            revision=SOURCE_REVISION,
            allow_patterns=[f"{SOURCE_PATH}/lerobot/**", f"{SOURCE_PATH}/lerobot__val/**"],
        )
    )
    result = prepare_mikasa_h8(
        snapshot / SOURCE_PATH / "lerobot",
        snapshot / SOURCE_PATH / "lerobot__val",
        output_train,
    )
    root = Path(output_train).parent
    (root / "provenance.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text(
        "# MIKASA InterceptGrabFast H8 teacher data\n\n"
        "Derived from the pinned H1 successful trajectories without recollection. "
        "Each anchor contains the next eight within-episode teacher actions; "
        "zero-padded tail positions are excluded by action_loss_mask.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-train", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_mikasa_h8_from_hub(args.output_train),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
