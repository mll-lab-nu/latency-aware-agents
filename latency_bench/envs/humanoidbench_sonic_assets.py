"""Explicitly prepare the pinned G1 scene assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from latency_bench.utils.paths import REPO_ROOT

SONIC_ASSETS_REPO_ID = "MLL-Lab/LAGEN-datasets"
SONIC_ASSETS_REVISION = "8678de04e6b59692d85c8e6e690c5ca482626857"


def prepare_g1_sonic_assets(cache_dir: Path) -> Path:
    """Download the pinned scene bundle."""

    snapshot = Path(
        snapshot_download(
            repo_id=SONIC_ASSETS_REPO_ID,
            repo_type="dataset",
            revision=SONIC_ASSETS_REVISION,
            allow_patterns=["assets/humanoidbench/g1-sonic/sonic_g1/**"],
            local_dir=cache_dir / "assets",
        )
    )
    asset_root = snapshot / "assets/humanoidbench/g1-sonic/sonic_g1"
    truck_source = (
        REPO_ROOT
        / "third_party/humanoid-bench/humanoid_bench/assets/tasks/truck.xml"
    )
    truck_target = asset_root / "tasks/truck.xml"
    truck_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(truck_source, truck_target)
    return asset_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    asset_root = prepare_g1_sonic_assets(args.cache_dir)
    print(json.dumps({"asset_root": str(asset_root)}, indent=2))


if __name__ == "__main__":
    main()
