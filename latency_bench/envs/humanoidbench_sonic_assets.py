"""Explicitly prepare the pinned SONIC G1 assets and decoder."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download, snapshot_download

SONIC_REPO_ID = "nvidia/GEAR-SONIC"
SONIC_REVISION = "1157dbafbae49ba17b8b53ace04cd8191b6a6697"
SONIC_DECODER_FILENAME = "sonic_v1_1/model_decoder.onnx"
SONIC_ENCODER_FILENAME = "sonic_v1_1/model_encoder.onnx"
SONIC_PLANNER_FILENAME = "planner_sonic.onnx"
SONIC_ASSETS_REPO_ID = "latency-sensitive-bench/g1-sonic-assets"
SONIC_ASSETS_REVISION = "7c0a6616c78106adf9fa54dfc40e56ea07dc7d5d"
BONES_SEED_REPO_ID = "bones-studio/seed"
BONES_SEED_REVISION = "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
BONES_SEED_ARCHIVE_FILENAME = "g1.tar.gz"
BONES_SEED_SKILL_CSVS = {
    "fast_run": "g1/csv/230907/jog_ff_loop_180_R_very_fast_002__A451.csv",
    "obstacle_jump": "g1/csv/231110/avoid_obstacle_jump_run_ff_180_R_002__A500.csv",
}


def prepare_g1_sonic_encoder(cache_dir: Path) -> Path:
    """Download the pinned SONIC v1.1 encoder."""

    return Path(
        hf_hub_download(
            repo_id=SONIC_REPO_ID,
            filename=SONIC_ENCODER_FILENAME,
            revision=SONIC_REVISION,
            local_dir=cache_dir / "encoder",
        )
    )


def prepare_g1_sonic_planner(cache_dir: Path) -> Path:
    """Download the pinned SONIC planner used by the retained FastTD3 run."""

    return Path(
        hf_hub_download(
            repo_id=SONIC_REPO_ID,
            filename=SONIC_PLANNER_FILENAME,
            revision=SONIC_REVISION,
            local_dir=cache_dir / "planner",
        )
    )


def prepare_g1_sonic_skill_references(cache_dir: Path) -> dict[str, Path]:
    """Download the pinned BONES-SEED archive and extract the selected skills."""

    bones_cache = cache_dir / "bones_seed"
    archive = Path(
        hf_hub_download(
            repo_id=BONES_SEED_REPO_ID,
            repo_type="dataset",
            filename=BONES_SEED_ARCHIVE_FILENAME,
            revision=BONES_SEED_REVISION,
            local_dir=bones_cache,
        )
    )
    csv_paths = {
        name: bones_cache / member
        for name, member in BONES_SEED_SKILL_CSVS.items()
    }
    missing = [name for name, path in csv_paths.items() if not path.exists()]
    if missing:
        remaining = {BONES_SEED_SKILL_CSVS[name]: name for name in missing}
        with tarfile.open(archive, mode="r|gz") as bundle:
            for member in bundle:
                if member.name not in remaining:
                    continue
                name = remaining.pop(member.name)
                csv_path = csv_paths[name]
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                with csv_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if not remaining:
                    break
    return csv_paths


def prepare_g1_sonic_assets(cache_dir: Path) -> tuple[Path, Path]:
    """Download the pinned scene bundle and dynamic-batch SONIC decoder."""

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
    hurdle_source = (
        Path(__file__).resolve().parents[2]
        / "third_party/humanoid-bench/humanoid_bench/assets/locomotion/generated_xml_hurdles.xml"
    )
    hurdle_target = asset_root / "locomotion/generated_xml_hurdles.xml"
    hurdle_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hurdle_source, hurdle_target)
    truck_source = hurdle_source.parents[1] / "tasks/truck.xml"
    truck_target = asset_root / "tasks/truck.xml"
    truck_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(truck_source, truck_target)

    decoder = Path(
        hf_hub_download(
            repo_id=SONIC_REPO_ID,
            filename=SONIC_DECODER_FILENAME,
            revision=SONIC_REVISION,
            local_dir=cache_dir / "decoder",
        )
    )
    dynamic_decoder = decoder.with_name("model_decoder_dynamic.onnx")
    if not dynamic_decoder.exists():
        model = onnx.load(decoder)
        for value_info in (*model.graph.input, *model.graph.output):
            batch = value_info.type.tensor_type.shape.dim[0]
            batch.ClearField("dim_value")
            batch.dim_param = "batch"
        onnx.save(model, dynamic_decoder)
    return asset_root, dynamic_decoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    asset_root, decoder_path = prepare_g1_sonic_assets(args.cache_dir)
    print(
        json.dumps(
            {"asset_root": str(asset_root), "decoder_path": str(decoder_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
