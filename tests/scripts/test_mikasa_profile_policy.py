"""MIKASA profile evaluation uses pinned inputs and an explicit checkpoint."""

import sys
from pathlib import Path

import yaml

from latency_bench.core.config import load_config
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
