"""Run the complete HAIC profile, native teacher, VLA, and eval pipeline."""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess

import yaml

from latency_bench.core.config import load_config


PROFILE_SCRIPT = Path("scripts/tasks/haic_pull_cart/profile.sh")
TRAIN_COMMAND_SCRIPT = Path("scripts/tasks/haic_pull_cart/train_latency_command.sh")
TRAIN_VLA_SCRIPT = Path("scripts/tasks/haic_pull_cart/train_vla.sh")
HAIC_SCRIPT = Path("third_party/HAIC/scripts/haic_vla.py")
def _profile_from_log(path: Path) -> Path:
    return Path(
        next(
            line.removeprefix("PROFILE_PATH=")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("PROFILE_PATH=")
        )
    )


def _native_checkpoint_from_log(path: Path) -> Path:
    return Path(
        next(
            line.partition("Saved checkpoint to ")[2]
            for line in path.read_text(encoding="utf-8").splitlines()
            if "Saved checkpoint to " in line
            and line.partition("Saved checkpoint to ")[2].endswith(
                "checkpoint_final.pt"
            )
        )
    ).resolve()


def _model_path_from_log(path: Path) -> Path:
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
    runtime_checkpoint: Path,
    profile_path: Path,
    method: str,
    inference_device: str,
    simulator_device: str,
    output_dir: Path,
    episodes: int,
    max_steps: int,
    inference_batch_size: int,
    latency_config: dict,
) -> None:
    config = load_config(source)
    config["executor"].update(deepcopy(latency_config["executor"]))
    config["executor"].update(
        {
            "mode": "simulated",
            "simulated_inference_pool": True,
            "inference_devices": [inference_device],
            "inference_batch_size": inference_batch_size,
        }
    )
    config["scheduler"] = deepcopy(latency_config["scheduler"])
    config["env"].update(
        {
            "runtime_checkpoint_path": str(runtime_checkpoint),
            "simulator_device": simulator_device,
        }
    )
    config["policy"].update(
        {
            "model_path": str(model_path),
            "device": inference_device,
        }
    )
    config["latency"] = deepcopy(latency_config["latency"])
    config["latency"]["method"] = method
    config["latency"]["profile_path"] = str(profile_path)
    config["latency"]["sync_cuda"] = False
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
        "--profile-path",
        type=Path,
        help="Use an existing profile instead of measuring one.",
    )
    parser.add_argument("--skip-native-train", action="store_true")
    parser.add_argument("--native-checkpoint", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    config = load_config(args.config)
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ | config["environment"]
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "third_party/HAIC")]
    )

    def run(stage: str, command: list[str], *, extra_env: dict[str, str] | None = None) -> Path:
        log = output / f"{stage}.log"
        print(f"{stage}: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command,
                cwd=root,
                env=environment | (extra_env or {}),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return log

    profile = args.profile_path
    if profile is None:
        profile_source = config["profile_source"]
        profile_command = ["bash", str(PROFILE_SCRIPT)]
        if profile_source == "checkpoint":
            profile_command.append(str(config["profile_model"]))
        elif profile_source != "hf_artifacts":
            raise ValueError(f"unsupported HAIC profile source: {profile_source}")
        profile_command.extend(config["profile_arguments"])
        profile = _profile_from_log(
            run(
                "profile",
                profile_command,
                extra_env={"HAIC_PYTHON": str(Path(config["haic_python"]).resolve())},
            )
        )
    profile = profile.resolve()

    latency = load_config(config["latency_config"])
    native_latency = deepcopy(latency)
    latency["latency"]["method"] = config["profile_latency_method"]
    latency["latency"]["profile_path"] = str(profile)
    latency_path = output / "latency.yaml"
    latency_path.write_text(yaml.safe_dump(latency, sort_keys=False), encoding="utf-8")

    native_output = Path(config["native_output_dir"]).resolve()
    if not args.skip_native_train and config["native_train"]:
        native_command = [
            "bash",
            str(TRAIN_COMMAND_SCRIPT),
            str(Path(config["teacher_checkpoint"]).resolve()),
            str(latency_path),
            f"hydra.run.dir={native_output}",
            *config["native_train_arguments"],
        ]
        native_log = run(
            "native_train",
            native_command,
            extra_env={
                "HAIC_PYTHON": str(Path(config["haic_python"]).resolve()),
                "WANDB_DIR": str(native_output),
            },
        )
        native_checkpoint = _native_checkpoint_from_log(native_log)
    else:
        native_checkpoint = (
            args.native_checkpoint.resolve()
            if args.native_checkpoint is not None
            else Path(config["native_checkpoint"]).resolve()
        )

    vla = config["vla"]
    vla_output = output / "vla"
    train_command = [
        "bash",
        str(TRAIN_VLA_SCRIPT),
        str(native_checkpoint),
        "--base-model",
        str(Path(config["base_model"]).resolve()),
        "--output-root",
        str(vla_output),
        "--policy-config",
        str(Path(config["policy_config"]).resolve()),
        "--latency-profile-path",
        str(profile),
        "--latency-config",
        str(latency_path),
        "--runtime-checkpoint",
        str(Path(config["teacher_checkpoint"]).resolve()),
        "--wandb-run-id",
        str(vla["run_id"]),
        "--start-round",
        str(vla["start_round"]),
        "--end-round",
        str(vla["end_round"]),
        "--inference-device",
        str(config["inference_device"]),
        "--inference-batch-size",
        str(vla["inference_batch_size"]),
    ]
    if vla["smoke"]:
        train_command.append("--smoke")
    vla_log = run(
        "vla",
        train_command,
        extra_env={"HAIC_PYTHON": str(Path(config["haic_python"]).resolve())},
    )

    model_path = _model_path_from_log(vla_log)

    inference_device = str(config["inference_device"])
    simulator_device = str(config["simulator_device"])
    for model_name, model in (
        ("baseline", Path(config["profile_model"]).resolve()),
        ("new", model_path),
    ):
        for method in ("zero", latency["latency"]["method"]):
            stage = f"eval_{model_name}_{method}"
            eval_output = output / stage
            eval_config = output / f"{stage}.yaml"
            _write_eval_config(
                Path(config["evaluation_config"]).resolve(),
                eval_config,
                model_path=model,
                runtime_checkpoint=Path(config["teacher_checkpoint"]).resolve(),
                profile_path=profile,
                method=method,
                inference_device=inference_device,
                simulator_device=simulator_device,
                output_dir=eval_output,
                episodes=config["eval_episodes"],
                max_steps=config["eval_max_steps"],
                inference_batch_size=config["eval_inference_batch_size"],
                latency_config=native_latency,
            )
            run(
                stage,
                [
                    str(Path(config["haic_python"]).resolve()),
                    str(HAIC_SCRIPT),
                    "latency-eval",
                    "--headless",
                    "--eval-config",
                    str(eval_config),
                ],
            )

    print(f"PROFILE_PATH={profile}")
    print(f"NATIVE_CHECKPOINT={native_checkpoint}")
    print(f"MODEL_PATH={model_path}")


if __name__ == "__main__":
    main()
