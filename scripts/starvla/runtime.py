"""Shared process and policy-input settings for StarVLA evaluation scripts."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess

from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.utils.paths import REPO_ROOT


ROOT = REPO_ROOT
STARVLA = ROOT / "third_party/starVLA"

TASK_INPUT_KEYS = {
    "humanoidbench_balance_simple": {
        "state_info_key": "gymnasium_state",
        "image_views_info_key": ENV_RAW_RGB_FRAME_STACK_INFO_KEY,
    },
    "mikasa_intercept_grab_fast": {
        "state_info_key": "mikasa_proprio",
        "image_views_info_key": "mikasa_image_views",
    },
}


def environment() -> dict[str, str]:
    return os.environ | {
        "PYTHONPATH": os.pathsep.join([
            str(ROOT), str(STARVLA),
            str(ROOT / "third_party/humanoid-bench"),
            str(ROOT / "third_party/sample-factory"),
        ]),
        "TOKENIZERS_PARALLELISM": "false",
        "MUJOCO_GL": "egl",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def run_command(command: list, log: Path, *, cwd: Path = ROOT, env: dict | None = None) -> None:
    command = [str(part) for part in command]
    log.parent.mkdir(parents=True, exist_ok=True)
    print(shlex.join(command), flush=True)
    print(f"Log: {log}", flush=True)
    with log.open("w") as output:
        subprocess.run(command, cwd=cwd, env=environment() if env is None else env,
                       stdout=output, stderr=subprocess.STDOUT, check=True)


def evaluation_command(args, config_path):
    """Launch the task's native environment with the shared evaluation driver."""
    if args.task == "balance":
        command = [args.task_python, "-m", "latency_bench.run", "--config", config_path]
    elif args.task in ("mikasa", "mikasa_h8", "mikasa_conditioned_h8"):
        command = [
            args.task_python,
            ROOT / "scripts/mikasa/evaluate.py",
            "latency-eval",
            "--eval-config",
            config_path,
        ]
    return command
