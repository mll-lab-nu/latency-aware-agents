"""Long-lived MIKASA orchestration contract: profile → teacher → issued data → VLA → eval."""

import sys
from pathlib import Path

import yaml

from latency_bench.core.config import load_config
from scripts.tasks.mikasa_intercept_grab_fast import run_latency_pipeline


def test_pipeline_passes_measured_profile_and_new_checkpoints_to_consumers(tmp_path, monkeypatch):
    config = load_config("configs/training/mikasa_latency_pipeline_smoke.yaml")
    config["output_dir"] = str(tmp_path / "run")
    native = load_config(config["latency_config"])
    native["latency"]["seed"] = 17
    native["executor"]["simulated_worker_capacity"] = 3
    native["scheduler"]["hold_policy"] = "noop"
    native_path = tmp_path / "native.yaml"
    native_path.write_text(yaml.safe_dump(native))
    config["latency_config"] = str(native_path)
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    profile = tmp_path / "measured-profile.json"
    commands = []

    def run(command, *, cwd, env, stdout, stderr, check):
        commands.append(command)
        if command[1] == "scripts/profile.sh":
            stdout.write(f"PROFILE_PATH={profile}\n")
        if "latency-eval" in command:
            assert cwd == Path(__file__).resolve().parents[2] / "third_party/starVLA"

    monkeypatch.setattr(run_latency_pipeline.subprocess, "run", run)
    monkeypatch.setattr(run_latency_pipeline, "resolve_model_artifacts", lambda spec, *, hf_artifacts: {
        "checkpoint_path": "/baseline/checkpoints/steps_5000_pytorch_model.pt",
        "model_config_path": "/baseline/config.full.yaml",
    })
    monkeypatch.setattr(sys, "argv", ["pipeline", "--config", str(config_path)])
    run_latency_pipeline.main()
    output = tmp_path / "run"
    latency = yaml.safe_load((output / "latency.yaml").read_text())
    assert latency["latency"]["profile_path"] == str(profile)
    assert str(output / "teacher/teacher.pt") in commands[2]
    assert [command[-1] for command in commands[4:6]] == ["train[:1]", "train[1:2]"]
    for method in ("zero", "iid"):
        baseline = yaml.safe_load((output / f"eval_baseline_{method}.yaml").read_text())
        assert baseline["policy"]["checkpoint_path"] == "/baseline/checkpoints/steps_5000_pytorch_model.pt"
        assert baseline["latency"]["method"] == method
        assert baseline["latency"]["profile_path"] == str(profile)
        assert baseline["latency"]["seed"] == 17
        assert baseline["executor"]["simulated_worker_capacity"] == 3
        assert baseline["scheduler"] == native["scheduler"]
        evaluation = yaml.safe_load((output / f"eval_latency_{method}.yaml").read_text())
        assert evaluation["latency"]["method"] == method
        assert evaluation["latency"]["profile_path"] == str(profile)
        assert evaluation["latency"]["seed"] == 17
        assert evaluation["executor"]["simulated_worker_capacity"] == 3
        assert evaluation["scheduler"] == native["scheduler"]
        assert evaluation["policy"]["checkpoint_path"].endswith(
            f"{config['vla']['run_id']}/checkpoints/steps_2_pytorch_model.pt"
        )
