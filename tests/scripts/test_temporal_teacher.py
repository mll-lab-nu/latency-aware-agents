"""Protect profile identity, the historical recipe, and publish completion gates."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latency_bench.core.config import load_config
from latency_bench.latency.profile_builder import build_profile_from_traces
from latency_bench.utils.io import read_json, write_json
from scripts.experiments.temporal_profiles import teacher


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("teacher_source")
    rng = np.random.default_rng(791)
    sources = []
    for session in range(3):
        path = root / f"session_{session}.jsonl"
        busy = np.exp(rng.normal(4 + .1 * session, .1, 48))
        overhead = np.exp(rng.normal(-1, .1, 48))
        import json
        path.write_text("".join(json.dumps({
            "obs_id": i, "episode_id": i // 16, "obs_env_step": i,
            "submitted": True, "worker_id": 0,
            "observation_to_action_latency_ms": float(b + o), "worker_service_time_ms": float(b),
        }) + "\n" for i, (b, o) in enumerate(zip(busy, overhead))))
        sources.append({"measure_session_id": session, "observation_attempts": path})
    for method in teacher.METHODS:
        destination = root / teacher.PROFILE_ROOT / "profiles" / method
        build_profile_from_traces(
            sessions=sources, output_dir=destination, model_id="openvla", gpu_class="1x-rtx3090",
            workload_id="flappy", instance_id="fixture", source_run_id=teacher.SOURCE_RUN,
            env_fps=30, metadata={"session_ids": [0, 1, 2], "fit_options": {"method": method}},
            method=method, hmm_max_iterations=20, block_mean_length=32,
        )
        # Match the HF cache layout: companions are symlinks outside this folder.
        blobs = root / "blobs" / method
        blobs.mkdir(parents=True)
        for path in destination.glob("*.json"):
            target = blobs / path.name
            path.rename(target)
            path.symlink_to(target)
    return root


@pytest.mark.parametrize("method", teacher.METHODS)
def test_prepare_downloads_only_own_profile_and_preserves_teacher_recipe(tmp_path, snapshot, monkeypatch, method):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    teacher.prepare(tmp_path, method, 42, wandb_entity="test-team")
    assert calls[0] == {"repo_id": teacher.PROFILE_REPO, "repo_type": "dataset",
                        "revision": teacher.PROFILE_REVISION,
                        "allow_patterns": [f"{teacher.PROFILE_ROOT}/profiles/{method}/*"]}
    cfg = load_config(tmp_path / "train.yaml")
    assert cfg["env"]["env_fps"] == cfg["env"]["obs_fps"] == 10
    assert cfg["env"]["frame_stack"] == 4 and cfg["env"]["action_history_decisions"] == 8
    assert cfg["env"]["max_episode_steps"] == 0
    assert cfg["executor"]["simulated_worker_capacity"] == 1
    assert cfg["training"]["train_for_env_steps"] == 25_000_000
    assert cfg["training"]["num_workers"] == 2 and cfg["training"]["num_envs_per_worker"] == 128
    assert cfg["training"]["batch_size"] == 4096 and cfg["training"]["rollout"] == 128
    assert cfg["training"]["encoder_mlp_layers"] == [512, 512]
    assert cfg["latency"]["method"] == method and not cfg["latency"]["add_latency_info"]
    assert cfg["latency"]["seed"] == 42
    assert cfg["backend"]["restart_behavior"] == "restart"
    assert cfg["logging"]["wandb_project"] == "flappy-teachers"
    assert cfg["logging"]["wandb_entity"] == "test-team"
    assert not (tmp_path / "profile/profile.json").is_symlink()
    if method == "block":
        assert read_json(tmp_path / "profile/latency_block_model.json")["mean_block_length"] == 32
    first = read_json(tmp_path / "protocol.json")
    teacher.prepare(tmp_path, method, 42, wandb_entity="test-team")
    assert read_json(tmp_path / "protocol.json") == first
    assert load_config(tmp_path / "train.yaml")["logging"]["wandb_name"] == cfg["logging"]["wandb_name"]
    with pytest.raises(ValueError, match="settings changed"):
        teacher.prepare(tmp_path, method, 43, wandb_entity="test-team")


def test_rejects_held_out_sessions_and_edited_cached_profile(tmp_path, snapshot, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(snapshot))
    teacher.prepare(tmp_path, "block", 42, wandb_entity="test-team")
    profile = tmp_path / "profile/profile.json"
    data = read_json(profile)
    data["provenance"]["session_ids"] = [0, 1, 2, 3, 4]
    write_json(profile, data)
    with pytest.raises(ValueError, match="fitting sessions"):
        teacher.validate_profile(profile, "block")
    with pytest.raises(ValueError, match="Local profile changed"):
        teacher.prepare(tmp_path, "block", 42, wandb_entity="test-team")


def test_startup_and_resume_reach_sample_factory_with_wandb(tmp_path, snapshot, monkeypatch):
    from training.common.checkpoints import patch_sample_factory_checkpoint_loader
    from training.common.config_adapter import apply_runtime_logging_config
    from training.flappy_sf.integration import SAMPLE_FACTORY_CONFIG_PARSER, build_cli_args_from_config
    from sample_factory.algo.learning.learner import Learner

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(snapshot))
    kwargs = {"wandb_entity": "test-team", "wandb_project": "teacher-check"}
    teacher.prepare(tmp_path, "hmm", 42, **kwargs)
    config = load_config(tmp_path / "train.yaml")
    cfg = SAMPLE_FACTORY_CONFIG_PARSER.parse_train(build_cli_args_from_config(config))
    apply_runtime_logging_config(cfg, config["logging"])
    assert cfg.with_wandb is True
    assert cfg.wandb_user == "test-team" and cfg.wandb_project == "teacher-check"
    assert cfg.wandb_unique_id == read_json(tmp_path / "job.json")["wandb_name"]
    run_id = cfg.wandb_unique_id
    assert cfg.restart_behavior == "restart"
    # Exercise the real patched loader at exactly the point shown in the crash.
    patch_sample_factory_checkpoint_loader()
    learner = Learner.__new__(Learner)
    learner.is_initialized = False
    learner.cfg = cfg
    learner.device = "cpu"
    restored = []
    learner._load_state = lambda state, load_progress: restored.append((state, load_progress))
    learner.load_from_checkpoint(0)
    assert restored == []

    checkpoint_dir = Path(read_json(tmp_path / "job.json")["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "checkpoint_000000001_4096.pth"
    state = {"train_step": 1, "env_steps": 4096}
    torch.save(state, checkpoint)
    teacher.prepare(tmp_path, "hmm", 42, **kwargs)
    config = load_config(tmp_path / "train.yaml")
    cfg = SAMPLE_FACTORY_CONFIG_PARSER.parse_train(build_cli_args_from_config(config))
    apply_runtime_logging_config(cfg, config["logging"])
    assert cfg.restart_behavior == "resume" and cfg.wandb_unique_id == run_id
    learner.cfg = cfg
    learner.load_from_checkpoint(0)
    assert restored == [(state, True)]
    # An unreadable selected checkpoint must still fail, not start over.
    checkpoint.write_bytes(b"invalid checkpoint")
    with pytest.raises(Exception):
        learner.load_from_checkpoint(0)
    checkpoint.rename(checkpoint_dir / "best_000000001_4096_reward_1.pth")
    with pytest.raises(FileNotFoundError, match="refusing to start fresh"):
        teacher.prepare(tmp_path, "hmm", 42, **kwargs)

    other = tmp_path / "new_run"
    teacher.prepare(other, "hmm", 42, **kwargs)
    assert load_config(other / "train.yaml")["logging"]["wandb_name"] != run_id


def test_export_rejects_short_training_then_packages_selected_best(tmp_path, snapshot, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: str(snapshot))
    teacher.prepare(tmp_path, "block", 42, wandb_entity="test-team")
    cfg = load_config(tmp_path / "train.yaml")
    job = read_json(tmp_path / "job.json")
    root = Path(job["train_artifact_dir"])
    checkpoints = root / "checkpoint_p0"
    checkpoints.mkdir(parents=True)
    latest = checkpoints / "checkpoint_000000010_25000000.pth"
    best = checkpoints / "best_000000009_24000000_reward_12.000.pth"
    # File names alone must not establish completion.
    torch.save({"env_steps": 10}, latest)
    torch.save({"env_steps": 24_000_000}, best)
    flat = {"seed": 42, "env_fps": 10, "obs_fps": 10, "frame_stack": 4,
            "action_history_decisions": 8, "simulated_worker_capacity": 1,
            "train_for_env_steps": 25_000_000, "use_rnn": False, "add_latency_info": False,
            "save_best_metric": "reward", "max_episode_steps": 0, "latency_type": "block",
            "latency_profile_path": cfg["latency"]["profile_path"]}
    write_json(root / "config.json", flat)
    (tmp_path / "train.log").write_text("Training log fixture\n")
    with pytest.raises(ValueError, match="incomplete"):
        teacher.export(tmp_path)
    assert not (tmp_path / "bundle/DONE").exists()
    with pytest.raises(FileNotFoundError):
        teacher.upload(tmp_path, teacher.TEACHER_REPO)
    torch.save({"env_steps": 25_001_984}, latest)
    teacher.export(tmp_path)
    evidence = read_json(tmp_path / "bundle/training_complete.json")
    assert evidence["latest_decision_transitions"] == 25_001_984
    assert evidence["best_decision_transitions"] == 24_000_000
    assert evidence["selected_checkpoint"] == f"checkpoint_p0/{best.name}"
    assert (tmp_path / "bundle/profile/latency_block_model.json").exists()
    assert read_json(tmp_path / "bundle/config.json") == flat
    calls = []
    destination_exists = True
    class FakeApi:
        def create_repo(self, *args, **kwargs):
            calls.append(("create", args, kwargs))
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="before")
        def get_paths_info(self, *args, **kwargs):
            return [SimpleNamespace(path="existing")] if destination_exists else []
        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))
            return SimpleNamespace(oid="after", commit_url="https://example.test/commit/after")
    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        teacher.upload(tmp_path, teacher.TEACHER_REPO)
    assert [call[0] for call in calls] == ["create"]
    destination_exists = False
    calls.clear()
    first = teacher.upload(tmp_path, teacher.TEACHER_REPO)
    assert calls[0][2] == {"repo_type": "model", "private": True, "exist_ok": True}
    assert calls[1][1]["folder_path"] == str(tmp_path / "bundle")
    assert calls[1][1]["parent_commit"] == "before"
    assert teacher.upload(tmp_path, teacher.TEACHER_REPO) == first
    assert len(calls) == 2
