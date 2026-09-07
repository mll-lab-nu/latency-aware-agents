"""Run the Parkour profile, command policy, issued-data, VLA, and eval stages."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml

from latency_bench.core.config import load_config


PARKOUR_SCRIPT = Path("third_party/Isaaclab_Parkour/scripts/rsl_rl/parkour_vla.py")
TRAIN_VLA_SCRIPT = Path("scripts/tasks/extreme_parkour_go2/train_vla.sh")
TRAIN_COMMAND_SCRIPT = Path("scripts/tasks/extreme_parkour_go2/train_latency_command.sh")


def _profile_from_log(path: Path) -> Path:
    return Path(
        next(
            line.removeprefix("PROFILE_PATH=")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("PROFILE_PATH=")
        )
    ).resolve()


def _native_checkpoint_from_log(path: Path) -> Path:
    return Path(
        next(
            line.removeprefix("NATIVE_CHECKPOINT=")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("NATIVE_CHECKPOINT=")
        )
    ).resolve()


def _model_from_log(path: Path) -> Path:
    return Path(
        next(
            line.removeprefix("MODEL_PATH=")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("MODEL_PATH=")
        )
    ).resolve()


def _write_eval_config(
    source: Path,
    destination: Path,
    *,
    model_path: Path,
    decoder_checkpoint: Path,
    latency_config: dict,
    method: str,
    inference_devices: list[str],
    output_dir: Path,
    episodes: int,
    max_steps: int,
    inference_batch_size: int,
) -> None:
    config = load_config(source)
    config["executor"] = {
        "mode": "simulated",
        "simulated_worker_capacity": latency_config["executor"]["simulated_worker_capacity"],
        "simulated_inference_pool": True,
        "inference_devices": inference_devices,
        "inference_batch_size": inference_batch_size,
    }
    config["env"].update(
        {
            "runtime_checkpoint_path": str(decoder_checkpoint),
            "simulator_device": "cuda:0",
        }
    )
    config["policy"].update(
        {
            "model_path": str(model_path),
            "device": inference_devices[0],
        }
    )
    config["latency"] = latency_config["latency"] | {"method": method}
    config["scheduler"] = latency_config["scheduler"]
    config["evaluation"].update(
        {
            "eval_episodes": episodes,
            "eval_parallel_envs": 1,
            "eval_max_steps": max_steps,
            "eval_deterministic": True,
        }
    )
    config["logging"]["output_dir"] = str(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--profile-path", type=Path, help="Use an existing profile instead of measuring one."
    )
    parser.add_argument("--skip-native-train", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    config = load_config(args.config)
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    parkour_python = Path(config["parkour_python"])
    teacher_checkpoint = Path(config["teacher_checkpoint"]).resolve()
    inference_devices = config["inference_devices"]
    environment = os.environ | config["environment"]
    environment["PATH"] = str(parkour_python.parent) + os.pathsep + environment["PATH"]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "third_party/Isaaclab_Parkour")]
    )

    def run(stage: str, command: list[str]) -> Path:
        log = output / f"{stage}.log"
        print(f"{stage}: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return log

    profile = args.profile_path
    if profile is None:
        profile_args = config["profile_arguments"]
        profile_command = [
            "bash",
            "scripts/profile.sh",
            "run",
            "--model",
            "official_gr00t",
            "--workload",
            "extreme_parkour_go2",
        ]
        if "--hf-artifacts" not in profile_args:
            profile_command.extend(["--model-path", str(config["profile_model"])])
        profile_command.extend(profile_args)
        profile = _profile_from_log(run("profile", profile_command))
    profile = profile.resolve()

    latency = load_config(config["latency_config"])
    latency["latency"]["method"] = config["native_latency_method"]
    latency["latency"]["profile_path"] = str(profile)
    latency_path = output / "latency.yaml"
    latency_path.write_text(yaml.safe_dump(latency, sort_keys=False), encoding="utf-8")

    if not args.skip_native_train and config["native_train"]:
        native_log = run(
            "native_train",
            [
                "bash",
                str(TRAIN_COMMAND_SCRIPT),
                str(teacher_checkpoint),
                str(latency_path),
                *config["native_train_arguments"],
            ],
        )
        native_checkpoint = _native_checkpoint_from_log(native_log)
    else:
        native_checkpoint = Path(config["native_checkpoint"]).resolve()

    collection = config["collection"]
    raw_dir = (output / collection["raw_dir"]).resolve()
    dataset_dir = (output / collection["dataset_dir"]).resolve()
    run(
        "collect",
        [
            str(parkour_python),
            str(PARKOUR_SCRIPT),
            "collect",
            "--headless",
            "--device",
            str(collection["simulation_device"]),
            "--checkpoint",
            str(teacher_checkpoint),
            "--command-checkpoint",
            str(native_checkpoint),
            "--latency-config",
            str(latency_path),
            "--output_dir",
            str(raw_dir),
            "--train_episodes",
            str(collection["train_episodes"]),
            "--val_episodes",
            str(collection["val_episodes"]),
            "--split_seed",
            str(collection["split_seed"]),
            "--num_envs",
            str(collection["num_envs"]),
            *(["--keep-failed"] if collection["keep_failed"] else []),
        ],
    )
    run(
        "prepare",
        [
            sys.executable,
            "scripts/official_gr00t/prepare_parkour_dataset.py",
            str(raw_dir),
            str(dataset_dir),
            "--force",
            "--video-shards",
        ],
    )

    vla = config["vla"]
    vla_log = run(
        "vla",
        [
            "bash",
            str(TRAIN_VLA_SCRIPT),
            str(teacher_checkpoint),
            "--base-model",
            str(config["base_model"]),
            "--bootstrap-dataset",
            str(dataset_dir),
            "--bootstrap-raw",
            str(raw_dir),
            "--output-root",
            str(output / vla["output_dir"]),
            "--policy-config",
            str(config["policy_config"]),
            "--parkour-python",
            str(parkour_python),
            "--latency-profile-path",
            str(profile),
            "--latency-config",
            str(latency_path),
            "--wandb-run-id",
            str(vla["run_id"]),
            "--bootstrap-updates",
            str(vla["bootstrap_updates"]),
            "--inference-batch-size",
            str(vla["inference_batch_size"]),
            "--num-gpus",
            str(vla["num_gpus"]),
            "--global-batch-size",
            str(vla["global_batch_size"]),
            "--start-round",
            str(vla["start_round"]),
            "--end-round",
            str(vla["end_round"]),
            *sum((["--inference-device", device] for device in inference_devices), []),
        ],
    )
    model_path = _model_from_log(vla_log)

    for model_name, model in (
        ("baseline", Path(config["profile_model"]).resolve()),
        ("new", model_path),
    ):
        for method in config["evaluation_methods"]:
            stage = f"eval_{model_name}_{method}"
            eval_output = output / stage
            eval_config = output / f"{stage}.yaml"
            _write_eval_config(
                Path(config["evaluation_config"]),
                eval_config,
                model_path=model,
                decoder_checkpoint=teacher_checkpoint,
                latency_config=latency,
                method=method,
                inference_devices=inference_devices,
                output_dir=eval_output,
                episodes=config["eval_episodes"],
                max_steps=config["eval_max_steps"],
                inference_batch_size=config["eval_inference_batch_size"],
            )
            run(
                stage,
                [
                    str(parkour_python),
                    str(PARKOUR_SCRIPT),
                    "latency-eval",
                    "--headless",
                    "--device",
                    str(collection["simulation_device"]),
                    "--checkpoint",
                    str(teacher_checkpoint),
                    "--eval-config",
                    str(eval_config),
                    "--output_dir",
                    str(eval_output),
                    "--inference_batch_size",
                    str(config["eval_inference_batch_size"]),
                    *sum((["--inference_device", device] for device in inference_devices), []),
                ],
            )

    print(f"PROFILE_PATH={profile}")
    print(f"NATIVE_CHECKPOINT={native_checkpoint}")
    print(f"MODEL_PATH={model_path}")


if __name__ == "__main__":
    main()
