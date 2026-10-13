"""Prepare H1 collection/conversion configs and finalize local provenance."""

import argparse
from copy import deepcopy
from pathlib import Path
import os


from latency_bench.core.config import load_config
from latency_bench.utils.io import read_json, write_json, write_yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "finalize"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latency-frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = config.pop("dataset")
    output = args.output.resolve()
    data = output / "dataset"
    checkpoint = args.checkpoint.resolve()
    latency_ms = args.latency_frames * 1000.0 / config["env"]["env_fps"]
    if args.stage == "prepare":
        if not args.package_only:
            config["policy"] = {"type": "mikasa_ppo", "source": "local",
                                "root": str(checkpoint.parent), "checkpoint": checkpoint.name}
            config["latency"]["fixed_latency_ms"] = latency_ms
            config["logging"]["output_dir"] = str(data)
            config["evaluation"]["episodes"] = settings["train_episodes"] + settings["val_episodes"]
            write_yaml(output / "collect.yaml", config)
            write_yaml(output / "latency.yaml", {key: deepcopy(config[key]) for key in (
                "env", "executor", "scheduler", "latency")})
        write_yaml(output / "convert.yaml", {"conversion": {
            "format": "lerobot", "task": "mikasa",
            "input_dir": str(data / "data_npz/intercept_grab_fast_vla_v0"),
            "output_dir": str(data / "lerobot"),
            "train_episodes": settings["train_episodes"], "val_episodes": settings["val_episodes"],
        }})
        return
    metadata = read_json(data / "data_npz/intercept_grab_fast_vla_v0/metadata.json")
    write_json(data / "provenance.json", {
        "task": "InterceptGrabFast-VLA-v0", "teacher_checkpoint": str(checkpoint),
        "latency_frames": args.latency_frames, "latency_ms": latency_ms,
        "env_fps": config["env"]["env_fps"], "gpu_id": os.environ["CUDA_VISIBLE_DEVICES"], "train_episodes": settings["train_episodes"],
        "val_episodes": settings["val_episodes"], "acceptance": "success_once",
        "collector_metadata": metadata, "action_labels": "issued H1 commands",
        "packaging": read_json(data / "packaging.json"),
    })
    print(f"OUTPUT_DIR={data}")


if __name__ == "__main__":
    main()
