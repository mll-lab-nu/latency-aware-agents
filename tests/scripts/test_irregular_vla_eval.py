"""Uploaded students must run through the existing hardware-timed evaluator."""
from argparse import Namespace
from pathlib import Path
import sys

import huggingface_hub
import pytest

from latency_bench.core.config import load_config, resolve_runtime_config
from latency_bench.policy.model_source import resolve_model_source
from scripts.experiments.memory_mock import eval_vla


@pytest.mark.parametrize("method", ["temporal", "ar1", "arma"])
def test_download_then_realtime_config_preserves_history_and_records_revisions(tmp_path, monkeypatch, method):
    snapshot = tmp_path / "model-cache" / "snapshots" / ("a" * 40)
    backbone = tmp_path / "backbone-cache" / eval_vla.BACKBONE_REVISION
    backbone.mkdir(parents=True)
    for name in eval_vla.REQUIRED_FILES:
        path = snapshot / method / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test artifact")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(backbone if kwargs["repo_id"] == eval_vla.BACKBONE_REPO else snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    args = Namespace(method=method, repo_id="test/models", revision="main", episodes=100,
                     max_steps=3600, seed=1000000, output_dir=tmp_path / "eval")
    path = eval_vla.prepare(args)
    assert len(calls) == 2
    assert all(p.startswith(method + "/") for p in calls[0]["allow_patterns"])
    assert f"{method}/{eval_vla.CHECKPOINT}" in calls[0]["allow_patterns"]
    assert calls[1]["revision"] == eval_vla.BACKBONE_REVISION
    # Exercise real model-source and runtime resolution: local paths must retain
    # HF provenance and override the training server's backbone location.
    config = resolve_model_source(load_config(path))
    resolve_runtime_config(config)
    assert len(calls) == 2  # No second model/backbone download during resolution.
    assert config["provenance"]["model_artifact"]["revision"] == "a" * 40
    assert config["policy"]["backbone_path"] == str(backbone)
    assert config["policy"]["root"] == str(snapshot / method)
    assert config["policy"]["state_source"] == "transport"
    assert config["policy"]["prompt_mode"] == "latency_neutral"
    assert config["policy"]["normalization_robot_type"] == "rl_games_flappy"
    assert config["env"]["action_history_decisions"] == 8
    assert config["env"]["frame_stack"] == 1
    assert config["env"]["env_fps"] == config["env"]["obs_fps"] == 10
    assert config["executor"]["mode"] == "realtime"
    assert config["executor"]["parallel_envs"] == 1
    assert config["executor"]["inference_devices"] == ["cuda:0"]
    assert not config["executor"]["inference_pool"]
    assert "latency" not in config and "profiling" not in config
    assert config["evaluation"]["episodes"] == 100
    assert config["evaluation"]["max_env_steps"] == 3600
    assert config["experiment"]["seed"] == 1000000
    with pytest.raises(FileExistsError, match="fresh EVAL_RUN_ID"):
        eval_vla.prepare(args)


def test_missing_upload_does_not_fall_back_to_another_method(tmp_path, monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshot")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    args = Namespace(method="ar1", repo_id="test/models", revision="main", episodes=2,
                     max_steps=3600, seed=1000000, output_dir=tmp_path / "eval")
    with pytest.raises(FileNotFoundError, match="Upload the completed ar1"):
        eval_vla.prepare(args)
    assert len(calls) == 1
    assert not args.output_dir.exists()


def test_main_delegates_to_existing_cli_and_propagates_failure(tmp_path, monkeypatch):
    path = tmp_path / "eval_config.yaml"
    monkeypatch.setattr(eval_vla, "prepare", lambda args: path)
    monkeypatch.setattr(sys, "argv", ["eval_vla", "--method", "temporal", "--output-dir", str(tmp_path)])
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        raise eval_vla.subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(eval_vla.subprocess, "run", run)
    with pytest.raises(eval_vla.subprocess.CalledProcessError):
        eval_vla.main()
    assert calls == [([sys.executable, "-m", "latency_bench", "eval", "--config", str(path)],
                      {"cwd": eval_vla.REPO_ROOT, "check": True})]
