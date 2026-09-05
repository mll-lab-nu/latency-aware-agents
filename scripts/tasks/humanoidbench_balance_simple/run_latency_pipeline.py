"""Run the Balance profile, FastTD3, dataset export, and VLA evaluation chain."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml

from latency_bench.core.config import load_config
from training.common.checkpoints import find_checkpoint


ROOT = Path(__file__).resolve().parents[3]
PROFILE_ENTRY = "scripts/tasks/humanoidbench_balance_simple/profile.sh"
TRAIN_ENTRY = "scripts/tasks/humanoidbench_balance_simple/train_small_policy.sh"
VLA_ENTRY = "scripts/tasks/humanoidbench_balance_simple/train_vla.sh"


def _run(
    stage: str,
    command: list[str],
    *,
    output_dir: Path,
    environment: dict[str, str],
) -> str:
    log_path = output_dir / f"{stage}.log"
    print(f"{stage}: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    return log_path.read_text(encoding="utf-8")


def _profile_path(profile_output: str) -> Path:
    matches = [
        line.removeprefix("PROFILE_PATH=")
        for line in profile_output.splitlines()
        if line.startswith("PROFILE_PATH=")
    ]
    return Path(next(iter(matches))).expanduser().resolve()


def _write_latency_config(
    base_path: str | Path,
    profile_path: Path,
    output_dir: Path,
    *,
    native_config: dict[str, Any],
) -> Path:
    config = deepcopy(native_config)
    latency_config = load_config(ROOT / Path(base_path))
    for section in ("executor", "latency", "scheduler"):
        if section in latency_config:
            if section in config:
                config[section].update(latency_config[section])
            else:
                config[section] = deepcopy(latency_config[section])
    config["latency"]["profile_path"] = str(profile_path)
    latency_path = output_dir / "latency.yaml"
    latency_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return latency_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--profile-path",
        type=Path,
        help="Use this existing profile instead of measuring the configured model.",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    os.chdir(ROOT)
    config: dict[str, Any] = load_config(config_path)
    output_dir = (ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ | config["environment"]

    if args.profile_path is None:
        profile_output = _run(
            "profile",
            [
                "bash",
                PROFILE_ENTRY,
                config["profile_model"],
                *config["profile_args"],
            ],
            output_dir=output_dir,
            environment=environment,
        )
        profile_path = _profile_path(profile_output)
    else:
        profile_path = args.profile_path.expanduser().resolve()
    print(f"PROFILE_PATH={profile_path}", flush=True)

    train_config_path = ROOT / Path(config["native_train_config"])
    train_config = load_config(train_config_path)
    latency_path = _write_latency_config(
        config["latency_config"],
        profile_path,
        output_dir,
        native_config=train_config,
    )
    run_name = config["native_train_run_name"]
    environment["BALANCE_TRAIN_CONFIG"] = str(train_config_path)
    environment["BALANCE_EVAL_CONFIG"] = str(
        ROOT / Path(config["native_eval_config"])
    )
    environment["BALANCE_TRAIN_DIR"] = str(
        ROOT / Path(train_config["backend"]["train_dir"])
    )
    environment["BALANCE_TRAIN_RUN_NAME"] = run_name
    environment["BALANCE_TASK_ROOT"] = str(output_dir)
    environment["BALANCE_VLA_OUTPUT_DIR"] = str(output_dir / "vla")
    environment["BALANCE_RUN_ID"] = config["vla_run_id"]
    if config["baseline_model_path"] is not None:
        environment["BALANCE_BASELINE_MODEL_PATH"] = str(
            ROOT / Path(config["baseline_model_path"])
        )
    environment["BALANCE_VLA_MAX_STEPS"] = str(config["vla_max_steps"])
    environment["BALANCE_VLA_STEP_INTERVAL"] = str(config["vla_step_interval"])
    environment["BALANCE_VLA_START_STEP"] = str(config["vla_start_step"])
    environment["BALANCE_VLA_NUM_GPUS"] = str(config["vla_num_gpus"])
    environment["BALANCE_VLA_GLOBAL_BATCH_SIZE"] = str(config["vla_global_batch_size"])
    collection = config["collection"]
    environment["BALANCE_EXPORT_MAX_EPISODES"] = str(collection["max_episodes"])
    environment["BALANCE_EXPORT_VAL_FRACTION"] = str(collection["val_fraction"])
    environment["BALANCE_EXPORT_NUM_ENVS"] = str(collection["num_envs"])
    environment["BALANCE_EXPORT_REPLAY_NUM_ENVS"] = str(
        collection["replay_num_envs"]
    )
    environment["BALANCE_EXPORT_IMAGE_WRITER_WORKERS"] = str(
        collection["image_writer_workers"]
    )
    environment["BALANCE_EXPORT_ROW_BUFFER_SIZE"] = str(collection["row_buffer_size"])
    environment["BALANCE_EXPORT_DEVICE"] = str(collection["device"])
    _run(
        "native_train",
        [
            "bash",
            TRAIN_ENTRY,
            "--run-name",
            run_name,
            "--latency-config",
            str(latency_path),
            "--restart-behavior",
            config["native_restart_behavior"],
        ],
        output_dir=output_dir,
        environment=environment,
    )

    teacher_checkpoint = find_checkpoint(
        train_config["backend"]["train_dir"],
        run_name,
        kind=config["native_checkpoint_kind"],
    )
    _run(
        "vla",
        ["bash", VLA_ENTRY, str(teacher_checkpoint), str(latency_path)],
        output_dir=output_dir,
        environment=environment,
    )
    print(f"TEACHER_CHECKPOINT={teacher_checkpoint}", flush=True)
    print(f"LATENCY_CONFIG={latency_path}", flush=True)


if __name__ == "__main__":
    main()
