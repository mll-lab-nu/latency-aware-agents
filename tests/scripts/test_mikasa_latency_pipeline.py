"""Long-lived MIKASA orchestration contract: pinned teacher → issued data → shared H1 training."""

import sys
from pathlib import Path

import yaml

from latency_bench.core.config import load_config
from scripts.tasks.mikasa_intercept_grab_fast import run_latency_pipeline
from scripts.tasks.mikasa_intercept_grab_fast import run_profile_policy


def test_matched_profile_evals_hold_last_chunk_action():
    for config_path in (
        "configs/training/latency/mikasa_qwenoft_rtx3090_eval.yaml",
        "configs/training/latency/mikasa_qwenpi_v3_rtx3090_eval.yaml",
        "configs/training/latency/mikasa_qwengr00t_rtx3090_eval.yaml",
    ):
        config = load_config(config_path)
        assert config["scheduler"] == {
            "ordering_policy": "latest_ready",
            "hold_policy": "hold",
            "hold_last_chunk_action": True,
        }


def test_profile_policy_launcher_materializes_pinned_profile_and_checkpoint(tmp_path, monkeypatch):
    """Long-lived native launch contract: pinned data and the chosen checkpoint."""
    config = load_config("configs/training/latency/mikasa_qwenoft_rtx3090_eval.yaml")
    config["latency"]["profile_path"] = str(tmp_path / "profile/profile.json")
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(config))
    snapshot = tmp_path / "snapshot"
    source = snapshot / config["profile_source"]["path"]
    source.mkdir(parents=True)
    (source / "profile.json").write_text('{"source_run_id":"pinned-source"}')
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"chosen checkpoint")
    output = tmp_path / "evaluation"

    def download(*, repo_id, repo_type, revision, allow_patterns):
        assert repo_id == config["profile_source"]["repo_id"]
        assert repo_type == "dataset"
        assert revision == config["profile_source"]["revision"]
        return str(snapshot)

    commands = []
    monkeypatch.setattr(run_profile_policy, "snapshot_download", download)
    monkeypatch.setattr(run_profile_policy.os, "execvpe", lambda executable, command, environment: commands.append(command))
    monkeypatch.setattr(sys, "argv", [
        "run_profile_policy", "--config", str(path), "--output-dir", str(output),
        "--checkpoint", str(checkpoint),
    ])
    monkeypatch.chdir(tmp_path)
    run_profile_policy.main()
    resolved = load_config(output / "config.yaml")
    assert Path(resolved["latency"]["profile_path"]).read_text() == (source / "profile.json").read_text()
    assert resolved["training"]["checkpoint"] == str(checkpoint)
    assert resolved["training"]["evaluate"] is True
    assert resolved["training"]["resume"] is None
    assert str(output / "config.yaml") in commands[0]


def test_pipeline_distills_pinned_teacher_with_shared_h1_training(tmp_path, monkeypatch):
    config = load_config("configs/training/mikasa_latency_pipeline.yaml")
    config["output_dir"] = str(tmp_path / "run")
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    native = load_config(config["latency_config"])
    snapshot = tmp_path / "snapshot"
    profile = snapshot / native["profile_source"]["path"]
    profile.mkdir(parents=True)
    (profile / "profile.json").write_text('{"source_run_id":"pinned-profile"}')
    teacher = tmp_path / "accepted.pt"
    commands = []

    def download_teacher(repo_id, filename, *, revision, local_dir):
        assert repo_id == config["teacher_source"]["repo_id"]
        assert filename == config["teacher_source"]["path"]
        assert revision == config["teacher_source"]["revision"]
        return str(teacher)

    def download_profile(repo_id, *, repo_type, revision, allow_patterns):
        assert repo_id == native["profile_source"]["repo_id"]
        assert revision == native["profile_source"]["revision"]
        assert repo_type == "dataset"
        return str(snapshot)

    def run(command, *, cwd, env, stdout, stderr, check):
        commands.append(command)
        assert env["WANDB_MODE"] == "online"
        if "--path-to-save-data" in command:
            Path(command[command.index("--path-to-save-data") + 1]).mkdir()

    monkeypatch.setattr(run_latency_pipeline, "hf_hub_download", download_teacher)
    monkeypatch.setattr(run_latency_pipeline, "snapshot_download", download_profile)
    monkeypatch.setattr(run_latency_pipeline.subprocess, "run", run)
    monkeypatch.setattr(run_latency_pipeline.shutil, "move", lambda *args: None)
    monkeypatch.setattr(run_latency_pipeline, "_write_mikasa_contract", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["pipeline", "--config", str(config_path)])
    run_latency_pipeline.main()
    output = tmp_path / "run"
    latency = yaml.safe_load((output / "latency.yaml").read_text())
    assert latency["latency"]["profile_path"] == str(output / "profile/profile.json")
    assert latency["latency"]["seed"] == 0
    assert commands[0][commands[0].index("--checkpoint") + 1] == str(teacher)
    assert [command[-1] for command in commands[2:4]] == ["train[:250]", "train[250:275]"]
    training = commands[4]
    assert training[2:4] == [str(output / "dataset"), str(teacher)]
    assert training[training.index("--micro-batch") + 1] == "64"
    assert training[training.index("--num-gpus") + 1] == "2"
    assert "--no-publish" in training
    assert len(commands) == 5
