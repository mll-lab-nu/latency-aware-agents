"""Fetch the pinned profile and run native MIKASA PPO training or evaluation."""

import argparse
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import snapshot_download
import yaml

from latency_bench.core.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Override the policy checkpoint in the config.")
    parser.add_argument("--wandb-project", help="Enable W&B in this project")
    parser.add_argument("--wandb-entity", help="Optional W&B account or team")
    args = parser.parse_args()
    config = load_config(args.config)
    config["training"]["track"] = args.wandb_project is not None
    config["training"]["wandb_project_name"] = args.wandb_project
    config["training"]["wandb_entity"] = args.wandb_entity
    output = args.output_dir.resolve()
    if args.checkpoint is not None:
        config["training"]["checkpoint"] = str(args.checkpoint.resolve())

    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    if config["latency"]["method"] in ("iid", "temporal"):
        source = config["profile_source"]
        snapshot = snapshot_download(
            repo_id=source["repo_id"], repo_type="dataset", revision=source["revision"],
            allow_patterns=[source["path"] + "/*"],
        )
        profile_dir = Path(config["latency"]["profile_path"]).parent
        shutil.copytree(Path(snapshot) / source["path"], profile_dir, dirs_exist_ok=True)

    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "config.yaml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    mikasa = root / "third_party/MIKASA-Robo"
    command = [
        sys.executable, "-u",
        str(mikasa / "mikasa_robo_suite/vla/dataset_collectors/get_dataset_collectors_ckpt.py"),
        "--env-id", "InterceptGrabFast-VLA-v0", "--latency-config", str(resolved),
        "--latency-output-dir", str(output),
    ]
    environment = os.environ | {"PYTHONPATH": os.pathsep.join([str(root), str(mikasa)])}
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
