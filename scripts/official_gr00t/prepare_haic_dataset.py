"""Finalize HAIC bootstrap or DAgger shards as an official GR00T dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from latency_bench.data.haic_dagger import (
    convert_haic_bootstrap_shards_to_lerobot,
    convert_haic_dagger_shards_to_lerobot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GR00T_PYTHON = REPO_ROOT / "third_party/Isaac-GR00T/.venv/bin/python"
MODALITY_CONFIG = "latency_bench/policy/official_gr00t_haic_modality.py"


def prepare_haic_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    force: bool,
    dagger: bool = False,
    round_index: int | None = None,
    expected_rows: int | None = None,
    normalization_source: Path | None = None,
    modality_config: str = MODALITY_CONFIG,
) -> dict:
    if dagger and normalization_source is None:
        raise TypeError("--normalization-source is required for --dagger")
    converter = (
        convert_haic_dagger_shards_to_lerobot
        if dagger
        else convert_haic_bootstrap_shards_to_lerobot
    )
    manifest = converter(
        raw_dir,
        output_dir,
        force=force,
        expected_rows=expected_rows,
        round_index=round_index,
    )
    dataset_dirs = [output_dir]
    val_dir = output_dir.with_name(f"{output_dir.name}__val")
    if val_dir.exists():
        dataset_dirs.append(val_dir)
    if dagger:
        source_stats = normalization_source / "meta/stats.json"
        for dataset_dir in dataset_dirs:
            shutil.copy2(source_stats, dataset_dir / "meta/stats.json")
        return manifest
    for dataset_dir in dataset_dirs:
        subprocess.run(
            [
                str(GR00T_PYTHON),
                "-m",
                "gr00t.data.stats",
                "--dataset-path",
                str(dataset_dir),
                "--embodiment-tag",
                "NEW_EMBODIMENT",
                "--modality-config-path",
                modality_config,
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dagger", action="store_true")
    parser.add_argument("--round-index", type=int)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--normalization-source", type=Path)
    parser.add_argument("--modality-config", default=MODALITY_CONFIG)
    args = parser.parse_args()
    manifest = prepare_haic_dataset(
        args.raw_dir.resolve(),
        args.output_dir.resolve(),
        force=args.force,
        dagger=args.dagger,
        round_index=args.round_index,
        expected_rows=args.expected_rows,
        normalization_source=(
            args.normalization_source.resolve()
            if args.normalization_source is not None
            else None
        ),
        modality_config=args.modality_config,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
