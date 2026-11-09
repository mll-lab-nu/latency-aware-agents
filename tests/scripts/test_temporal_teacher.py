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

PROFILE_CASES = [(name, method) for name, source in teacher.PROFILE_SETS.items()
                 for method in source["methods"]]


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
    for profile_set, method in PROFILE_CASES:
        source = teacher.PROFILE_SETS[profile_set]
        destination = root / source["root"] / "profiles" / method
        build_profile_from_traces(
            sessions=sources, output_dir=destination, model_id="openvla", gpu_class="1x-rtx3090",
            workload_id="flappy", instance_id="fixture", source_run_id=source["source_run"],
            env_fps=30, metadata={"session_ids": [0, 1, 2], "fit_options": {"method": method}},
            method=method, hmm_max_iterations=20, block_mean_length=32,
        )
        write_json(destination / "hardware.json", {"gpu_class": "1x-rtx3090"})
        (destination / "DONE").touch()
        # Match the HF cache layout: companions are symlinks outside this folder.
        blobs = root / "blobs" / profile_set / method
        blobs.mkdir(parents=True)
        for path in destination.glob("*.json"):
            target = blobs / path.name
            path.rename(target)
            path.symlink_to(target)
    return root


@pytest.mark.parametrize(("profile_set", "method"), PROFILE_CASES)
def test_prepare_downloads_only_own_profile_and_preserves_teacher_recipe(tmp_path, snapshot, monkeypatch, profile_set, method):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot / kwargs["filename"])
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    kwargs = {"wandb_entity": "test-team", "profile_set": profile_set}
    teacher.prepare(tmp_path, method, 42, **kwargs)
    source = teacher.PROFILE_SETS[profile_set]
    assert calls[0] == {"repo_id": source["repo_id"], "repo_type": "dataset",
                        "revision": source["revision"],
                        "filename": f"{source['root']}/profiles/{method}/profile.json"}
    expected_files = {"profile.json", "hardware.json", "latency_profile.png", "DONE"}
    expected_files.update({"latency_burst_model.json", "latency_distribution.json"}
                          if method == "temporal" else {f"latency_{method}_model.json"})
    assert {call["filename"] for call in calls} == {
        f"{source['root']}/profiles/{method}/{name}" for name in expected_files}
    assert {path.name for path in (tmp_path / "profile").iterdir()} == expected_files
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
    teacher.prepare(tmp_path, method, 42, **kwargs)
    assert read_json(tmp_path / "protocol.json") == first
    assert load_config(tmp_path / "train.yaml")["logging"]["wandb_name"] == cfg["logging"]["wandb_name"]
    with pytest.raises(ValueError, match="settings changed"):
        teacher.prepare(tmp_path, method, 43, **kwargs)


@pytest.mark.parametrize("method", ["temporal", "ar1", "arma"])
def test_irregular_profiles_reuse_recipe_and_sample_at_teacher_fps(tmp_path, snapshot, monkeypatch, method):
    from latency_bench.core.clock import ms_to_raw_frames
    from latency_bench.latency.samplers import build_profile_sampler, temporal_profile_sampler_for_reset
    from training.common.latency_config import sample_factory_latency_params_from_flat_config
    from training.flappy_sf.integration import SAMPLE_FACTORY_CONFIG_PARSER, build_cli_args_from_config

    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(snapshot / kwargs["filename"]))
    old, new = tmp_path / "old", tmp_path / "new"
    teacher.prepare(old, "temporal", 42, wandb_entity="test-team")
    teacher.prepare(new, method, 42, wandb_entity="test-team", profile_set="irregular_20min")
    previous, current = load_config(old / "train.yaml"), load_config(new / "train.yaml")
    for section in ("env", "executor", "training", "scheduler", "policy"):
        assert current[section] == previous[section]
    assert current["evaluation"]["episodes"] == previous["evaluation"]["episodes"] == 100
    assert current["evaluation"]["max_env_steps"] == previous["evaluation"]["max_env_steps"] == 3600
    assert current["evaluation"]["suites"][0]["latency"] == {
        "method": method, "profile_path": current["latency"]["profile_path"], "seed": 42}

    cfg = SAMPLE_FACTORY_CONFIG_PARSER.parse_train(build_cli_args_from_config(current))
    params = sample_factory_latency_params_from_flat_config(cfg, env_fps=cfg.env_fps)
    assert params.profile_method == method and params.simulated_worker_capacity == 1
    sampler = build_profile_sampler(params.profile_path, method=params.profile_method, seed=42,
                                    env_fps=cfg.env_fps, obs_fps=cfg.obs_fps)
    assert read_json(params.profile_path)["env_fps"] == 30
    assert sampler.clock.env_fps == sampler.clock.obs_fps == 10
    a, b = [temporal_profile_sampler_for_reset(sampler, 17) for _ in range(2)]
    for _ in range(32):
        sample = a.sample({"worker_slot": 0, "episode_id": 0})
        other = b.sample({"worker_slot": 0, "episode_id": 0})
        assert sample == other
        assert sample.latency_raw_frames == ms_to_raw_frames(sample.latency_ms, 10)

    with pytest.raises(ValueError, match="Wrong source profile"):
        teacher.validate_profile(old / "profile/profile.json", "temporal", profile_set="irregular_20min")
    with pytest.raises(ValueError, match="not available"):
        teacher.download_profile(new, "hmm", profile_set="irregular_20min")


def test_rejects_held_out_sessions_and_edited_cached_profile(tmp_path, snapshot, monkeypatch):
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(snapshot / kwargs["filename"]))
    teacher.prepare(tmp_path, "block", 42, wandb_entity="test-team")
    profile = tmp_path / "profile/profile.json"
    data = read_json(profile)
    data["provenance"]["session_ids"] = [0, 1, 2, 3, 4]
    write_json(profile, data)
    with pytest.raises(ValueError, match="fitting sessions"):
        teacher.validate_profile(profile, "block")
    with pytest.raises(ValueError, match="Local profile changed"):
        teacher.prepare(tmp_path, "block", 42, wandb_entity="test-team")


@pytest.mark.parametrize("missing", ["profile.json", "latency_block_model.json"])
def test_partial_cache_surfaces_download_failure_and_can_retry(tmp_path, snapshot, monkeypatch, missing):
    from huggingface_hub.errors import LocalEntryNotFoundError

    def unavailable_file(**kwargs):
        if Path(kwargs["filename"]).name == missing:
            raise LocalEntryNotFoundError(f"Not cached and Hub unavailable: {missing}")
        return str(snapshot / kwargs["filename"])

    monkeypatch.setattr("huggingface_hub.hf_hub_download", unavailable_file)
    with pytest.raises(LocalEntryNotFoundError, match="Hub unavailable"):
        teacher.prepare(tmp_path, "block", 42, wandb_entity="test-team")
    assert not (tmp_path / "profile").exists()
    assert not (tmp_path / "protocol.json").exists()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(snapshot / kwargs["filename"]))
    teacher.prepare(tmp_path, "block", 42, wandb_entity="test-team")
    assert (tmp_path / "profile/latency_block_model.json").is_file()


@pytest.mark.parametrize(("profile_set", "method"), [("original", "hmm"), ("irregular_20min", "ar1"), ("irregular_20min", "arma")])
def test_startup_and_resume_reach_sample_factory_with_wandb(tmp_path, snapshot, monkeypatch, profile_set, method):
    from training.common.checkpoints import patch_sample_factory_checkpoint_loader
    from training.common.config_adapter import apply_runtime_logging_config
    from training.flappy_sf.integration import SAMPLE_FACTORY_CONFIG_PARSER, build_cli_args_from_config
    from sample_factory.algo.learning.learner import Learner

    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(snapshot / kwargs["filename"]))
    kwargs = {"wandb_entity": "test-team", "wandb_project": "teacher-check", "profile_set": profile_set}
    teacher.prepare(tmp_path, method, 42, **kwargs)
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
    teacher.prepare(tmp_path, method, 42, **kwargs)
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
        teacher.prepare(tmp_path, method, 42, **kwargs)

    other = tmp_path / "new_run"
    teacher.prepare(other, method, 42, **kwargs)
    assert load_config(other / "train.yaml")["logging"]["wandb_name"] != run_id


@pytest.mark.parametrize(("profile_set", "method"), [("original", "block"), ("irregular_20min", "temporal"), ("irregular_20min", "ar1"), ("irregular_20min", "arma")])
def test_export_rejects_short_training_then_packages_selected_best(tmp_path, snapshot, monkeypatch, profile_set, method):
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(snapshot / kwargs["filename"]))
    teacher.prepare(tmp_path, method, 42, wandb_entity="test-team", profile_set=profile_set)
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
            "save_best_metric": "reward", "max_episode_steps": 0,
            "latency_type": "profile_sample" if method == "temporal" else method,
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
    companion = "latency_burst_model.json" if method == "temporal" else f"latency_{method}_model.json"
    assert (tmp_path / "bundle/profile" / companion).exists()
    source = teacher.PROFILE_SETS[profile_set]
    readme = (tmp_path / "bundle/README.md").read_text()
    assert source["repo_id"] in readme and source["revision"] in readme
    launcher = Path(teacher.__file__).parent.parent / source["launcher_dir"] / f"train_{method}.sh"
    assert (tmp_path / "bundle/code" / launcher.name).read_bytes() == launcher.read_bytes()
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
    assert first["path_in_repo"].startswith(f"flappy/{method}/{source['source_run']}/seed_42/")
    assert calls[0][2] == {"repo_type": "model", "private": True, "exist_ok": True}
    assert calls[1][1]["folder_path"] == str(tmp_path / "bundle")
    assert calls[1][1]["parent_commit"] == "before"
    assert teacher.upload(tmp_path, teacher.TEACHER_REPO) == first
    assert len(calls) == 2
