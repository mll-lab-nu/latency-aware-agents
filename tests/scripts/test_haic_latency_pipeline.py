"""Orchestration contract for the HAIC latency pipeline."""

import sys

import yaml

from latency_bench.core.config import load_config
from scripts.tasks.haic_pull_cart import run_latency_pipeline


def test_pipeline_wires_profile_native_teacher_and_both_eval_models(
    tmp_path, monkeypatch
):
    config = load_config("configs/training/haic_latency_pipeline.yaml")
    config.update(
        {
            "output_dir": str(tmp_path / "run"),
            "profile_model": str(tmp_path / "baseline"),
            "teacher_checkpoint": str(tmp_path / "privileged.pt"),
            "base_model": str(tmp_path / "base-model"),
            "policy_config": "configs/examples/haic/official_gr00t_eval.yaml",
            "evaluation_config": "configs/examples/haic/official_gr00t_eval.yaml",
            "latency_config": "configs/training/latency/haic.yaml",
            "haic_python": str(tmp_path / "haic-python"),
            "native_output_dir": str(tmp_path / "run/native"),
            "eval_episodes": 1,
            "eval_max_steps": 2,
            "eval_inference_batch_size": 1,
        }
    )
    latency_config = load_config("configs/training/latency/haic.yaml")
    latency_config["latency"].update({"seed": 17})
    latency_config["executor"]["simulated_worker_capacity"] = 3
    latency_config["scheduler"].update(
        {"hold_policy": "drop", "ordering_policy": "oldest_ready"}
    )
    latency_config_path = tmp_path / "latency.yaml"
    latency_config_path.write_text(yaml.safe_dump(latency_config), encoding="utf-8")
    config["latency_config"] = str(latency_config_path)
    config["vla"].update({"run_id": "haic-test", "end_round": 1})
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    profile = tmp_path / "profile.json"
    native = tmp_path / "run/native/wandb/offline-run/files/checkpoint_final.pt"
    commands = []

    def run(command, *, cwd, env, stdout, stderr, check):
        commands.append(command)
        if command[1] == str(run_latency_pipeline.PROFILE_SCRIPT):
            stdout.write(f"PROFILE_PATH={profile}\n")
        elif command[1] == str(run_latency_pipeline.TRAIN_SMALL_POLICY_SCRIPT):
            native.parent.mkdir(parents=True)
            native.write_bytes(b"native")
            stdout.write(f"Saved checkpoint to {native}\n")
        elif command[1] == str(run_latency_pipeline.TRAIN_VLA_SCRIPT):
            model = tmp_path / "run/vla/training/haic-test/checkpoint-500"
            model.mkdir(parents=True)
            stdout.write(f"MODEL_PATH={model}\n")

    monkeypatch.setattr(run_latency_pipeline.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["pipeline", "--config", str(config_path)])

    run_latency_pipeline.main()

    output = tmp_path / "run"
    assert len(commands) == 7
    assert commands[0][:3] == ["bash", str(run_latency_pipeline.PROFILE_SCRIPT), str(tmp_path / "baseline")]
    assert commands[1][:2] == ["bash", str(run_latency_pipeline.TRAIN_SMALL_POLICY_SCRIPT)]
    assert str(tmp_path / "privileged.pt") in commands[1]
    assert str(output / "latency.yaml") in commands[1]
    assert commands[2][:2] == ["bash", str(run_latency_pipeline.TRAIN_VLA_SCRIPT)]
    assert str(native) in commands[2]
    assert "--latency-config" in commands[2]
    assert str(profile) in commands[2]
    assert [command[2] for command in commands[3:]] == [
        "latency-eval",
        "latency-eval",
        "latency-eval",
        "latency-eval",
    ]
    for model_name in ("baseline", "new"):
        for method in ("zero", "temporal"):
            evaluation = yaml.safe_load(
                (output / f"eval_{model_name}_{method}.yaml").read_text()
            )
            assert evaluation["latency"]["method"] == method
            assert evaluation["latency"]["profile_path"] == str(profile)
            assert evaluation["env"]["runtime_checkpoint_path"] == str(
                tmp_path / "privileged.pt"
            )
    new_eval = yaml.safe_load((output / "eval_new_zero.yaml").read_text())
    assert new_eval["policy"]["model_path"].endswith(
        "vla/training/haic-test/checkpoint-500"
    )
    assert new_eval["latency"]["seed"] == 17
    assert new_eval["executor"]["simulated_worker_capacity"] == 3
    assert new_eval["scheduler"] == latency_config["scheduler"]
