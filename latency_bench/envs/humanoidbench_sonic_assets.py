"""Explicitly prepare the pinned SONIC G1 assets and decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from huggingface_hub import hf_hub_download, snapshot_download

SONIC_REPO_ID = "nvidia/GEAR-SONIC"
SONIC_REVISION = "1157dbafbae49ba17b8b53ace04cd8191b6a6697"
SONIC_PLANNER_FILENAME = "planner_sonic.onnx"
SONIC_ENCODER_FILENAME = "sonic_v1_1/model_encoder.onnx"
SONIC_DECODER_FILENAME = "sonic_v1_1/model_decoder.onnx"
SONIC_ASSETS_REPO_ID = "latency-sensitive-bench/g1-sonic-assets"
SONIC_ASSETS_REVISION = "7c0a6616c78106adf9fa54dfc40e56ea07dc7d5d"


def prepare_g1_sonic_reference_assets(cache_dir: Path) -> tuple[Path, Path]:
    """Download the pinned SONIC planner and v1.1 encoder."""

    planner = Path(
        hf_hub_download(
            repo_id=SONIC_REPO_ID,
            filename=SONIC_PLANNER_FILENAME,
            revision=SONIC_REVISION,
            local_dir=cache_dir / "planner",
        )
    )
    encoder = Path(
        hf_hub_download(
            repo_id=SONIC_REPO_ID,
            filename=SONIC_ENCODER_FILENAME,
            revision=SONIC_REVISION,
            local_dir=cache_dir / "encoder",
        )
    )
    return planner, encoder


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
    return snapshot / "sonic_g1", dynamic_decoder


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
