"""Local teacher selection, historical collection budgets, and portable VLA datasets."""
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from helpers import save_test_image, write_rollout_fixture
from latency_bench.data.action_layouts import action_spec_for_env
from latency_bench.data.prompts import BASE_PROMPTS
from latency_bench.data.rollout_probe import new_probe_state, write_probe_state
from latency_bench.utils.io import read_json, read_yaml, write_json_atomic
from scripts.experiments.temporal_profiles import rollout, teacher
from test_temporal_teacher import PROFILE_CASES, snapshot


@pytest.fixture
def local_bundle(tmp_path, snapshot):
    def build(method, profile_set="original"):
        source = teacher.PROFILE_SETS[profile_set]
        bundle = tmp_path / "teachers" / profile_set / method / "bundle"
        bundle.mkdir(parents=True)
        shutil.copytree(snapshot / source["root"] / "profiles" / method, bundle / "profile")
        checkpoints = bundle / "checkpoint_p0"
        checkpoints.mkdir()
        (checkpoints / "best_000000010_24000000_reward_100.pth").touch()
        (checkpoints / "checkpoint_000000020_25000000.pth").touch()
        write_json_atomic(bundle / "training_complete.json", {
            "status": "training_complete", "target_decision_transitions": 25_000_000,
            "latest_decision_transitions": 25_000_000,
            "selected_checkpoint": "checkpoint_p0/best_000000010_24000000_reward_100.pth",
            "latest_checkpoint": "checkpoint_p0/checkpoint_000000020_25000000.pth",
        })
        write_json_atomic(bundle / "config.json", {
            "env": "latency_flappy", "seed": 42, "latency_seed": 42,
            "env_fps": 10, "obs_fps": 10, "frame_stack": 4, "action_history_decisions": 8,
            "simulated_worker_capacity": 1, "latency_profile_worker_slot": 0,
            "hold_policy": "one_frame_then_noop", "ordering_policy": "latest_ready",
            "add_latency_info": False, "use_gpu_render": True,
            "latency_type": "profile_sample" if method == "temporal" else method,
            "latency_profile_path": "/old/training/machine/profile.json",
        })
        protocol = {"settings": {"method": method, "profile_source": {
            "repo_id": source["repo_id"], "repo_type": "dataset", "revision": source["revision"],
            "path": f"{source['root']}/profiles/{method}",
        }}}
        if profile_set != "original":
            protocol["profile_set"] = profile_set
        write_json_atomic(bundle / "protocol.json", protocol)
        (bundle / "DONE").touch()
        return bundle
    return build


@pytest.mark.parametrize(("profile_set", "method"), PROFILE_CASES)
def test_local_best_and_own_profile_preserve_recipe(tmp_path, local_bundle, monkeypatch, profile_set, method):
    def no_download(*args, **kwargs):
        pytest.fail("Local collection must not download checkpoints or profiles")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", no_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", no_download)
    bundle = local_bundle(method, profile_set)
    output = tmp_path / "rollout"
    rollout.prepare(output, bundle, method, profile_set=profile_set)
    cfg = read_yaml(output / "collect.yaml")
    assert cfg["policy"]["checkpoint"] == read_json(bundle / "training_complete.json")["selected_checkpoint"]
    assert cfg["policy"]["root"] == str(bundle)
    assert cfg["policy"]["deterministic"] is True
    assert cfg["latency"]["method"] == method
    assert cfg["latency"]["profile_path"] == str(output / "profile/profile.json")
    assert cfg["executor"]["mode"] == "simulated"
    assert cfg["executor"]["parallel_envs"] == cfg["collection"]["replay_num_envs"] == 256
    assert cfg["evaluation"] == {"episodes": 200, "max_env_steps": 7200}
    assert cfg["env"]["env_fps"] == cfg["env"]["obs_fps"] == 10
    assert cfg["collection"]["filter_preset"] == "none"
    assert cfg["collection"]["two_pass"] is True and cfg["collection"]["context_window"] == 1
    assert cfg["collection"]["val_fraction"] == .1 and cfg["collection"]["split_seed"] == 0
    assert cfg["latency"]["seed"] == cfg["experiment"]["seed"] == 42
    assert read_json(output / "profile/profile.json") == read_json(bundle / "profile/profile.json")
    before = read_json(output / "protocol.json")
    assert before["settings"]["initial_episodes"] == 200
    assert before["settings"]["episode_increment"] == 50
    assert before["settings"]["min_train_rows"] == 320_000
    if profile_set == "original":
        # Earlier completed collections predate the profile-set selector.
        before.pop("profile_set")
        write_json_atomic(output / "protocol.json", before)
    rollout.prepare(output, bundle, method, profile_set=profile_set)
    assert read_json(output / "protocol.json") == before
    with pytest.raises(ValueError, match="settings changed"):
        rollout.prepare(output, bundle, method, max_steps=3600, profile_set=profile_set)
    other = "hmm" if method != "hmm" else "block"
    with pytest.raises(ValueError, match="does not match"):
        rollout.prepare(tmp_path / "wrong_method", bundle, other, profile_set=profile_set)


def test_new_temporal_requires_new_run_and_resume_preserves_bundled_profile(tmp_path, local_bundle):
    old = local_bundle("temporal")
    new = local_bundle("temporal", "irregular_20min")
    output = tmp_path / "rollout"
    with pytest.raises(ValueError, match="Wrong source profile"):
        rollout.prepare(output, old, "temporal", profile_set="irregular_20min")
    assert not output.exists()
    with pytest.raises(ValueError, match="Wrong source profile"):
        rollout.prepare(output, new, "temporal")
    rollout.prepare(output, new, "temporal", profile_set="irregular_20min")
    profile = output / "profile/profile.json"
    edited = read_json(profile)
    edited["source_run_id"] = teacher.SOURCE_RUN
    write_json_atomic(profile, edited)
    with pytest.raises(ValueError, match="Local profile changed"):
        rollout.prepare(output, new, "temporal", profile_set="irregular_20min")


def write_replay(raw, specs, image):
    action_spec = action_spec_for_env("flappy")
    splits = {"train": [], "val": []}
    for spec in specs:
        history = np.zeros((8, 3), dtype=np.float32)
        for step, action in enumerate((1, 0, 1)):
            splits[spec["split"]].append({
                "image": str(image), "env_name": "flappy", "episode_idx": spec["episode_idx"],
                "decision_step": step, "split": spec["split"], "seed": spec["seed"],
                "prompt": BASE_PROMPTS["flappy"], "latency_raw_frames": 1, "latency_ms": 60.,
                action_spec.reward_field: .1, "action_id": action,
                "action_text": action_spec.labels[action], "action": [float(action == 0), float(action == 1)],
                "state": history.reshape(-1).tolist(),
            })
            history[:-1] = history[1:]
            history[-1] = [1, action, 0]
    write_rollout_fixture(raw, splits["train"], splits["val"], action_spec=action_spec)
    write_json_atomic(raw / "metadata.json", {
        "env_name": "flappy", "env_fps": 10, "obs_fps": 10, "obs_stride_raw_frames": 1,
        "action_history_decisions": 8, "action_layout": action_spec.layout,
        "action_labels": list(action_spec.labels), "reward_field": action_spec.reward_field,
        "rows_unit": "decision_step", "train_rows": len(splits["train"]), "val_rows": len(splits["val"]),
        "train_episodes": sum(s["split"] == "train" for s in specs),
        "val_episodes": sum(s["split"] == "val" for s in specs),
    })


@pytest.mark.parametrize(("profile_set", "method"), [
    ("original", "block"), ("irregular_20min", "temporal"),
    ("irregular_20min", "ar1"), ("irregular_20min", "arma"),
])
def test_quota_probe_replay_and_vla_conversion_then_upload_retry(tmp_path, local_bundle, monkeypatch, profile_set, method):
    bundle = local_bundle(method, profile_set)
    output = tmp_path / "rollout"
    rollout.prepare(output, bundle, method, episodes=10, min_train_rows=50, profile_set=profile_set)
    image = tmp_path / "frame.png"
    save_test_image(image, 12)
    calls = []

    def fake_collection(config):
        calls.append(deepcopy(config))
        raw = Path(config["logging"]["output_dir"])
        episodes = config["evaluation"]["episodes"]
        if config["collection"]["probe_only"]:
            state = new_probe_state()
            state.update(next_attempt_idx=episodes, attempted_episodes=episodes)
            # Finish order differs from seed order, as it does with batched environments.
            state["accepted_specs"] = [{
                "episode_idx": i, "seed": 42 + i, "split": "train", "episode_length": 3,
                "episode_raw_frames": 3, "episode_raw_return": .3, "termination_reason": "terminated",
            } for i in reversed(range(episodes))]
            write_probe_state(raw, state)
        else:
            specs = read_json(raw / "probe_state.json")["accepted_specs"]
            assert [s["seed"] for s in specs] == list(range(42, 42 + episodes))
            write_replay(raw, specs, image)

    monkeypatch.setattr("latency_bench.data.rollout.collect_rollouts", fake_collection)
    rollout.collect(output)
    assert [(c["evaluation"]["episodes"], c["collection"]["probe_only"]) for c in calls] == [
        (10, True), (60, True), (60, False)]
    assert read_json(output / "history_audit.json")["checked_rows"] == 180
    assert read_json(output / "quality_report.json")["train_episodes"] == 54
    rollout.export(output)
    source = teacher.PROFILE_SETS[profile_set]
    readme = (output / "README.md").read_text()
    assert source["repo_id"] in readme and source["revision"] in readme
    launcher = Path(teacher.__file__).parent.parent / source["launcher_dir"] / f"collect_{method}.sh"
    assert (output / "code" / launcher.name).read_bytes() == launcher.read_bytes()
    manifest = read_json(output / "lerobot/manifest.json")
    assert manifest["frames"] == 162 and manifest["validation_frames"] == 18
    assert manifest["state_normalization"] == {"min": [0.] * 24, "max": [1.] * 24}
    rows = pq.read_table(output / "lerobot/data/chunk-000/episode_000000.parquet").to_pylist()
    assert rows[0]["observation.state"] == [0.] * 24
    assert rows[1]["observation.state"] == [0.] * 21 + [1., 1., 0.]
    assert rows[0]["action"] == [0., 1.] and rows[1]["action"] == [1., 0.]
    assert rows[0]["observation.image"]["bytes"].startswith(b"\x89PNG")
    modality = read_json(output / "lerobot/meta/modality.json")
    assert modality["state"]["game_state"]["end"] == 24
    assert modality["action"]["button"]["end"] == 2
    # Reusing the collected dataset must not call the GPU execution boundary again.
    rollout.collect(output)
    rollout.export(output)
    assert len(calls) == 3
    # Upload/retry uses the saved dataset provenance, without the teacher bundle.
    shutil.rmtree(bundle)

    uploads = []
    class Api:
        def create_repo(self, repo, **kwargs):
            assert kwargs == {"repo_type": "dataset", "private": True, "exist_ok": True}
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="before")
        def get_paths_info(self, *args, **kwargs):
            return []
        def upload_folder(self, **kwargs):
            uploads.append(kwargs)
            if len(uploads) == 1:
                raise PermissionError("HF token needs write permission")
            return SimpleNamespace(oid="after", commit_url="https://example.test/commit/after")
    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    with pytest.raises(PermissionError):
        rollout.upload(output, rollout.DATASET_REPO)
    assert (output / "DONE").exists() and not (output / "upload_receipt.json").exists()
    receipt = rollout.upload(output, rollout.DATASET_REPO)
    assert receipt["revision"] == "after" and receipt["repo_type"] == "dataset"
    assert receipt["path_in_repo"].startswith(f"flappy/{method}/{source['source_run']}/seed_42/")
    assert uploads[-1]["parent_commit"] == "before"
    assert "lerobot/**" in uploads[-1]["allow_patterns"]
    assert not any("checkpoint" in pattern for pattern in uploads[-1]["allow_patterns"])
    assert rollout.upload(output, rollout.DATASET_REPO) == receipt and len(uploads) == 2


def test_incomplete_bundle_or_dataset_cannot_be_published(tmp_path, local_bundle):
    bundle = local_bundle("temporal")
    completed = read_json(bundle / "training_complete.json")
    completed["latest_decision_transitions"] = 1000
    write_json_atomic(bundle / "training_complete.json", completed)
    with pytest.raises(ValueError, match="completed training"):
        rollout.prepare(tmp_path / "rollout", bundle, "temporal")
    with pytest.raises(FileNotFoundError):
        rollout.upload(tmp_path / "rollout", rollout.DATASET_REPO)
