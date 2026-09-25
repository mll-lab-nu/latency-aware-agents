"""Explicitly prepare the pinned G1 scene assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from latency_bench.utils.paths import REPO_ROOT

SONIC_ASSETS_REPO_ID = "placeholder/g1-sonic-assets"
SONIC_ASSETS_REVISION = "7c0a6616c78106adf9fa54dfc40e56ea07dc7d5d"


def prepare_g1_sonic_assets(cache_dir: Path) -> Path:
    """Download the pinned scene bundle."""

    snapshot = Path(
        snapshot_download(
            repo_id=SONIC_ASSETS_REPO_ID,
            repo_type="dataset",
            revision=SONIC_ASSETS_REVISION,
            allow_patterns=["sonic_g1/**"],
            local_dir=cache_dir / "assets",
        )
    )
    asset_root = snapshot / "sonic_g1"
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
