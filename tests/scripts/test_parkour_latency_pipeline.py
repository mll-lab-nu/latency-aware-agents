"""Orchestration contract for the Parkour latency pipeline."""

import sys
import os
from pathlib import Path

import yaml

from latency_bench.core.config import load_config
from scripts.tasks.extreme_parkour_go2 import run_latency_pipeline


def test_pipeline_consumes_native_command_checkpoint_and_fresh_dataset(
    tmp_path, monkeypatch
):
    config = load_config("configs/training/parkour_latency_pipeline.yaml")
    config.update(
        {
            "output_dir": str(tmp_path / "run"),
            "profile_model": str(tmp_path / "profile-model"),
            "teacher_checkpoint": str(tmp_path / "teacher.pt"),
            "native_checkpoint": str(tmp_path / "native.pt"),
            "base_model": str(tmp_path / "base-model"),
            "parkour_python": str(tmp_path / "parkour-python"),
            "native_train_arguments": ["--max_iterations", "2"],
            "eval_episodes": 1,
            "eval_max_steps": 2,
            "eval_inference_batch_size": 1,
        }
    )
    config["collection"].update({"num_envs": 1, "train_episodes": 1, "val_episodes": 0, "keep_failed": True})
    native = load_config(config["latency_config"])
    native["latency"]["seed"] = 17
    native["executor"]["simulated_worker_capacity"] = 3
    native["scheduler"]["hold_policy"] = "noop"
    native_path = tmp_path / "native.yaml"
    native_path.write_text(yaml.safe_dump(native))
    config["latency_config"] = str(native_path)
    config["vla"].update(
        {"output_dir": "vla", "end_round": 0, "bootstrap_updates": 1, "inference_batch_size": 2}
    )
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    profile = tmp_path / "measured-profile.json"
    native_dir = tmp_path / "native-run"
    native_checkpoint = native_dir / "model_1.pt"
    native_checkpoint.parent.mkdir()
    native_checkpoint.write_bytes(b"native")
    model_path = tmp_path / "vla-model"
    commands = []

    def run(command, *, cwd, env, stdout, stderr, check):
        commands.append(command)
        if command[1] == "scripts/profile.sh":
            stdout.write(f"PROFILE_PATH={profile}\n")
        elif command[1] == str(run_latency_pipeline.TRAIN_COMMAND_SCRIPT):
            assert env["PATH"].split(os.pathsep)[0] == str(Path(config["parkour_python"]).parent)
            stdout.write(
                f"NATIVE_CHECKPOINT={native_checkpoint}\n"
            )
        elif command[1] == "scripts/official_gr00t/prepare_parkour_dataset.py":
            dataset_dir = command[3]
            (tmp_path / "run/collection/lerobot").mkdir(parents=True)
        elif command[1] == str(run_latency_pipeline.TRAIN_VLA_SCRIPT):
            stdout.write(f"MODEL_PATH={model_path}\n")

    monkeypatch.setattr(run_latency_pipeline.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["pipeline", "--config", str(config_path)])

    run_latency_pipeline.main()

    output = tmp_path / "run"
    assert len(commands) == 9
    assert commands[0][1:5] == ["scripts/profile.sh", "run", "--model", "official_gr00t"]
    assert commands[1][0:2] == ["bash", str(run_latency_pipeline.TRAIN_COMMAND_SCRIPT)]
    assert commands[2][0] == str(config["parkour_python"])
    assert commands[2][1] == str(run_latency_pipeline.PARKOUR_SCRIPT)
    assert commands[2][commands[2].index("--checkpoint") + 1] == str(
        tmp_path / "teacher.pt"
    )
    assert commands[2][commands[2].index("--command-checkpoint") + 1] == str(
        native_checkpoint
    )
    assert commands[2][commands[2].index("--latency-config") + 1] == str(
        output / "latency.yaml"
    )
    assert "--keep-failed" in commands[2]
    assert commands[3][1] == "scripts/official_gr00t/prepare_parkour_dataset.py"
    assert str(output / "collection/raw") in commands[3]
    assert str(output / "collection/lerobot") in commands[3]
    assert commands[4][0:2] == ["bash", str(run_latency_pipeline.TRAIN_VLA_SCRIPT)]
    assert str(output / "collection/raw") in commands[4]
    assert str(output / "collection/lerobot") in commands[4]
    assert str(tmp_path / "teacher.pt") in commands[4]
    assert "--latency-profile-path" in commands[4]
    assert commands[4][commands[4].index("--latency-config") + 1] == str(output / "latency.yaml")
    assert str(profile) in commands[4]
    assert commands[4][commands[4].index("--inference-batch-size") + 1] == "2"
    assert commands[4][commands[4].index("--num-gpus") + 1] == "2"
    assert commands[4][commands[4].index("--global-batch-size") + 1] == "128"
    for command in commands[5:]:
        assert command[1] == str(run_latency_pipeline.PARKOUR_SCRIPT)
        assert str(tmp_path / "teacher.pt") in command

    latency = yaml.safe_load((output / "latency.yaml").read_text())
    assert latency["latency"]["method"] == "temporal"
    assert latency["latency"]["profile_path"] == str(profile)
    for model_name, expected_model in (
        ("baseline", tmp_path / "profile-model"),
        ("new", model_path),
    ):
        for method in ("zero", "temporal"):
            evaluation = yaml.safe_load(
                (output / f"eval_{model_name}_{method}.yaml").read_text()
            )
            assert evaluation["latency"]["method"] == method
            assert evaluation["latency"]["profile_path"] == str(profile)
            assert evaluation["latency"]["seed"] == 17
            assert evaluation["executor"]["simulated_worker_capacity"] == 3
            assert evaluation["scheduler"] == native["scheduler"]
            assert evaluation["env"]["runtime_checkpoint_path"] == str(
                tmp_path / "teacher.pt"
            )
            assert evaluation["policy"]["model_path"] == str(expected_model)
