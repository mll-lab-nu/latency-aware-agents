#!/usr/bin/env python3
"""Materialize latency and dataset configs for conditioned H8 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

from latency_bench.core.config import load_config
from latency_bench.utils.paths import REPO_ROOT as ROOT
from scripts.starvla.train_task import eval_config, mikasa_h8_profile_latency
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
    parser.add_argument("--collection-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evaluation-run", type=Path)
    parser.add_argument("--evaluation-config", type=Path)
    parser.add_argument("--vla-python", type=Path)
    args = parser.parse_args()
    if args.evaluation_run is not None:
        settings = load_config(args.evaluation_config)
        training = load_config(args.evaluation_run / "bootstrap.yaml")
        evaluation_args = SimpleNamespace(**settings["args"], condition=args.condition,
            vla_python=args.vla_python, backbone_path=Path(training["framework"]["qwenvl"]["base_vlm"]))
        config = eval_config(evaluation_args, args.evaluation_run / "inference", args.output.parent / "development",
                             episodes=settings["episodes"], eval_latency_frames=FIXED_LATENCIES[args.condition])
        args.output.write_text(yaml.safe_dump(config, sort_keys=False))
        return
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
    if args.collection_config is not None:
        recipe = load_config(args.collection_config)
        checkpoint = args.checkpoint.resolve()
        directory = args.output.parent
        prefix = ({"mode": "profile_p95", "profile_path": config["latency"]["profile_path"],
                   "profile_revision": config["profile_source"]["revision"]}
                  if args.condition == "profile-latency"
                  else {"mode": "fixed_frames", "frames": FIXED_LATENCIES[args.condition]})
        recipe["policy"] = {"type": "mikasa_ppo", "source": "local", "root": str(checkpoint.parent),
                            "checkpoint": checkpoint.name, "action_prefix": prefix}
        recipe["latency"] = config["latency"]
        recipe["scheduler"] = config["scheduler"]
        recipe["logging"]["output_dir"] = str(directory / "raw")
        recipe["collection"].update(condition=args.condition,
                                     model=args.model if args.condition == "profile-latency" else "shared")
        (directory / "collect.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
        (directory / "convert.yaml").write_text(yaml.safe_dump({"conversion": {
            "format": "lerobot", "input_dir": str(directory / "raw"), "output_dir": str(directory / "lerobot")}},
            sort_keys=False))


if __name__ == "__main__":
    main()
