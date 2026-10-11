"""Run matched noop and random controls from a generated Flappy best-eval YAML."""

from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from latency_bench.core.config import load_config
from latency_bench.utils.io import write_yaml


CONTROL_POLICIES = {
    "noop": {"type": "constant", "action": "noop"},
    "random": {"type": "random", "actions": ["noop", "flap"]},
}


def write_control_configs(eval_config_path: Path, output_root: Path) -> dict[str, Path]:
    formal = load_config(eval_config_path)
    paths = {}
    for control, policy in CONTROL_POLICIES.items():
        config = deepcopy(formal)
        config["policy"] = deepcopy(policy)
        config["evaluation"]["episode_seed_offset"] = 104729
        config["logging"]["output_dir"] = str(output_root / control)
        for key in list(config["logging"]):
            if key.startswith("wandb_"):
                del config["logging"][key]
        path = output_root / f"{control}.yaml"
        write_yaml(path, config)
        paths[control] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control", choices=(*CONTROL_POLICIES, "both"), default="both")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    paths = write_control_configs(args.eval_config, args.output_root)
    controls = CONTROL_POLICIES if args.control == "both" else (args.control,)
    for control in controls:
        path = paths[control]
        print(f"{control}: {path}", flush=True)
        if not args.prepare_only:
            subprocess.run([sys.executable, "-m", "latency_bench", "eval", "--config", str(path)], check=True)


if __name__ == "__main__":
    main()
