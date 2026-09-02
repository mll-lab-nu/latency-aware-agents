"""Finalize Parkour episode shards as an official GR00T LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from latency_bench.data.action_layouts import GO2_PARKOUR_TERRAIN_LATENT_ACTION_SPEC
from latency_bench.data.hf_parquet import write_parquet_splits_from_shards
from latency_bench.data.official_gr00t import (
    convert_canonical_rollout_to_official_gr00t_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GR00T_PYTHON = REPO_ROOT / "third_party/Isaac-GR00T/.venv/bin/python"
MODALITY_CONFIG = "latency_bench/policy/official_gr00t_parkour_modality.py"


def prepare_parkour_dataset(
    raw_dir: Path, output_dir: Path, *, force: bool
) -> dict:
    write_parquet_splits_from_shards(
        raw_dir,
        action_spec=GO2_PARKOUR_TERRAIN_LATENT_ACTION_SPEC,
        keep_dataset=False,
    )
    manifest = convert_canonical_rollout_to_official_gr00t_dataset(
        raw_dir,
        output_dir,
        action_dim=32,
        action_horizon=1,
        modality_config=MODALITY_CONFIG,
        embodiment_tag="NEW_EMBODIMENT",
        action_layout=GO2_PARKOUR_TERRAIN_LATENT_ACTION_SPEC.layout,
        action_labels=GO2_PARKOUR_TERRAIN_LATENT_ACTION_SPEC.labels,
        force=force,
    )
    for dataset_dir in (output_dir, output_dir.with_name(f"{output_dir.name}__val")):
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
                MODALITY_CONFIG,
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
    args = parser.parse_args()
    manifest = prepare_parkour_dataset(
        args.raw_dir.resolve(), args.output_dir.resolve(), force=args.force
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
