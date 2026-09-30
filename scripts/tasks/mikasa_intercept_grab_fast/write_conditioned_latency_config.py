#!/usr/bin/env python3
"""Materialize one self-contained latency config for conditioned H8 collection."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from latency_bench.core.config import load_config
from latency_bench.utils.paths import REPO_ROOT as ROOT
from scripts.starvla.train_task import mikasa_h8_profile_latency
from scripts.tasks.mikasa_intercept_grab_fast.fixed_latency import FIXED_LATENCIES


PROFILE_CONFIGS = {
    "qwenoft": "configs/training/latency/mikasa_qwenoft_rtx3090_scratch.yaml",
    "qwenpi_v3": "configs/training/latency/mikasa_qwenpi_v3_rtx3090_scratch.yaml",
    "qwengr00t": "configs/training/latency/mikasa_qwengr00t_rtx3090_scratch.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("profile-latency", *FIXED_LATENCIES), required=True)
    parser.add_argument("--model", choices=tuple(PROFILE_CONFIGS), default="qwenoft")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(
        ROOT / (
            PROFILE_CONFIGS[args.model]
            if args.condition == "profile-latency"
            else "configs/training/latency/mikasa_fixed_2f.yaml"
        )
    )
    if args.condition == "profile-latency":
        config["latency"] = mikasa_h8_profile_latency(args.model)
    else:
        config["latency"]["fixed_latency_ms"] = FIXED_LATENCIES[args.condition] * 50.0
    config["scheduler"]["hold_last_chunk_action"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
