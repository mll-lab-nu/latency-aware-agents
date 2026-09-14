#!/usr/bin/env python3
"""Prepare an existing canonical LeRobot export for StarVLA."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq

from latency_bench.data.starvla_tasks import (
    StarVLATaskSpec,
    read_task_contract,
    task_spec,
    write_task_contract,
)


def _parquet_files(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "data").glob("*/*.parquet"))


def _canonical_modality(dataset_dir: Path, spec: StarVLATaskSpec) -> None:
    modality_path = dataset_dir / "meta" / "modality.json"
    modality = json.loads(modality_path.read_text(encoding="utf-8"))
    state_entries = list(modality["state"].values())
    action_entries = list(modality["action"].values())
    video_entries = list(modality["video"].values())
    state_meta = state_entries[0]
    action_meta = action_entries[0]
    if state_meta["end"] - state_meta["start"] != spec.state_dim:
        raise ValueError(f"{spec.name} state metadata does not have {spec.state_dim} values")
    if action_meta["end"] - action_meta["start"] != spec.action_dim:
        raise ValueError(f"{spec.name} action metadata does not have {spec.action_dim} values")
    if len(video_entries) != len(spec.video_keys):
        raise ValueError(
            f"{spec.name} expected {len(spec.video_keys)} video keys, got {len(video_entries)}"
        )

    def canonical_entry(entry: dict[str, Any], *, end: int, original_key: str) -> dict[str, Any]:
        result = dict(entry)
        result.update({"start": 0, "end": end, "original_key": original_key})
        return result

    canonical = {
        "state": {
            "proprio": canonical_entry(
                state_meta, end=spec.state_dim, original_key="observation.state"
            )
        },
        "action": {
            "target": canonical_entry(action_meta, end=spec.action_dim, original_key=spec.action_key)
        },
        "video": {
            key.removeprefix("video."): entry
            for key, entry in zip(spec.video_keys, video_entries)
        },
        "annotation": modality["annotation"],
    }
    modality_path.write_text(json.dumps(canonical, indent=2) + "\n", encoding="utf-8")


def _validate_rows(dataset_dir: Path, spec: StarVLATaskSpec) -> None:
    expected = {
        "observation.state": (spec.state_dim,),
        spec.action_key: ((spec.action_horizon, spec.action_dim) if spec.action_horizon > 1 else (spec.action_dim,)),
        **spec.auxiliary_shape_map(),
    }
    for path in _parquet_files(dataset_dir):
        for row in pq.read_table(path).to_pylist():
            for key, shape in expected.items():
                if key not in row:
                    raise KeyError(f"{key!r} is missing from {path}")
                actual_shape = np.asarray(row[key]).shape
                if actual_shape != shape:
                    raise ValueError(
                        f"{spec.name} field {key!r} must have shape {shape}, got {actual_shape} in {path}"
                    )


def _rewrite_dataset(dataset_dir: Path, spec: StarVLATaskSpec) -> None:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info["features"]
    if tuple(features["observation.state"]["shape"]) != (spec.state_dim,):
        raise ValueError(f"{spec.name} info metadata has the wrong state shape")
    if tuple(features[spec.action_key]["shape"]) != ((spec.action_horizon, spec.action_dim) if spec.action_horizon > 1 else (spec.action_dim,)):
        raise ValueError(f"{spec.name} info metadata has the wrong action shape")
    _validate_rows(dataset_dir, spec)
    _canonical_modality(dataset_dir, spec)


def _write_manifest(dataset_dir: Path, spec: StarVLATaskSpec, contract: dict[str, Any]) -> None:
    manifest_path = dataset_dir / "manifest.json"
    manifest = {
            "format": f"latency_bench_starvla_h{spec.action_horizon}_v1",
            "task": spec.name,
            "robot_type": spec.robot_type,
            "state_dim": spec.state_dim,
            "action_dim": spec.action_dim,
            "action_horizon": spec.action_horizon,
            "vla_fps": spec.fps,
            "task_contract": "task_contract.json",
            "state_key": contract["state_key"],
            "action_key": contract["action_key"],
            "video_keys": contract["video_keys"],
            "normalization": contract["normalization"],
        }
    for key in ("prompt", "state_labels", "action_labels", "auxiliary"):
        if key in contract:
            manifest[key] = contract[key]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def prepare_lerobot(
    task: str,
    source_train: Path | str,
    source_val: Path | str | None,
    output_train: Path | str,
    normalization_source: Path | str | None = None,
) -> dict[str, Any]:
    """Copy and prepare canonical LeRobot train/validation datasets once."""
    spec = task_spec(task)
    source_train = Path(source_train)
    output_train = Path(output_train)
    shutil.copytree(source_train, output_train)
    output_dirs = [output_train]
    if source_val is not None:
        output_val = output_train.with_name(f"{output_train.name}__val")
        shutil.copytree(Path(source_val), output_val)
        output_dirs.append(output_val)

    for dataset_dir in output_dirs:
        _rewrite_dataset(dataset_dir, spec)

    train_source = normalization_source
    write_task_contract(output_train, spec=spec, normalization_source=train_source)
    train_contract = read_task_contract(output_train)
    _write_manifest(output_train, spec, train_contract)
    if len(output_dirs) == 2:
        output_val = output_dirs[1]
        write_task_contract(
            output_val,
            spec=spec,
            normalization_source=train_source or output_train,
        )
        _write_manifest(output_val, spec, read_task_contract(output_val))

    return {
        "format": f"latency_bench_starvla_h{spec.action_horizon}_v1",
        "task": spec.name,
        "train": str(output_train),
        "validation": str(output_dirs[1]) if len(output_dirs) == 2 else None,
        "task_contract": str(output_train / "task_contract.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("balance", "haic", "mikasa", "parkour"), required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-val", type=Path)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--normalization-source", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_lerobot(
        args.task,
        args.source_train.expanduser().resolve(),
        args.source_val.expanduser().resolve() if args.source_val is not None else None,
        args.output_train.expanduser().resolve(),
        args.normalization_source.expanduser().resolve()
        if args.normalization_source is not None
        else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
