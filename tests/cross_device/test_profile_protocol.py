import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import sys
from threading import Event

import pytest

import latency_bench.cross_device.runner as runner
from latency_bench.cross_device.artifacts import (
    profile_dir,
    profile_ref,
    run_dir,
    run_ref,
    simulation_seed_dir,
)
from latency_bench.cross_device.profile_resolver import (
    profile_refs_from_repo_files,
    resolve_remote_profile,
)
from latency_bench.cross_device.protocol import main
from latency_bench.cross_device.runner import _profile_lock, _replace_current
from latency_bench.cross_device.runtime_configs import simulation_config
from latency_bench.cross_device.spec import load_profile_spec
from latency_bench.cross_device.sync import download_profile, publish_run_and_profile


def test_profile_coordinates_define_every_artifact_path(tmp_path):
    profile = profile_ref("openvla", "2x-rtx4090", "flappy", "instance_abc")
    run = run_ref(profile, "run_1")

    assert profile_dir(tmp_path, profile) == tmp_path / "profiles" / profile
    assert run_dir(tmp_path, run) == tmp_path / "runs" / run
    assert simulation_seed_dir(tmp_path, profile, "run_1", "temporal", 7) == (
        tmp_path / "simulations" / profile / "run_1" / "temporal" / "7"
    )
    assert profile_refs_from_repo_files(
        [
            f"profiles/{profile.as_posix()}/DONE",
            f"profiles/{profile.as_posix()}/profile.json",
            "profiles/too/short/DONE",
        ]
    ) == [profile]


def test_registry_has_one_openvla_flappy_preset():
    spec = load_profile_spec("configs/profiles.yaml", "openvla", "flappy")

    assert spec["session_count"] == 5
    assert spec["n_episodes"] == 100
    assert spec["calibrate_env_fps"] is True
    assert spec["artifact_root"] == "results/profiling"


def test_registry_has_balance_official_gr00t_preset():
    spec = load_profile_spec(
        "configs/profiles.yaml", "official_gr00t", "g1_sonic_balance_simple"
    )

    assert spec["session_count"] == 5
    assert spec["n_episodes"] == 100
    assert spec["evaluation_seed"] == 0
    assert spec["simulation_parallel_envs"] == 1
    assert spec["calibrate_env_fps"] is False
    assert spec["simulation_env"] == {}
    assert spec["checkpoint_path"].endswith("checkpoint-10000")
    assert spec["gpu_models"]["rpbzzz6"]["aliases"] == ["NVIDIA RPBZZZ6"]


def test_deployment_collection_preserves_cuda_device_selector(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(
        runner,
        "collect_hardware_fingerprint",
        lambda: {
            "gpu": {"topology": None},
            "gpus": [
                {"uuid": "GPU-b", "name": "NVIDIA RPBZZZ6"},
                {"uuid": "GPU-a", "name": "NVIDIA RPBZZZ6"},
            ],
        },
    )

    deployment, fingerprint = runner._collect_deployment(
        {"gpu_models": {"rpbzzz6": {"aliases": ["NVIDIA RPBZZZ6"]}}}
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert [(gpu["uuid"], gpu["cuda_index"]) for gpu in deployment["gpus"]] == [
        ("GPU-a", 1),
        ("GPU-b", 0),
    ]
    assert fingerprint["gpu_deployment"] == deployment


def test_official_gr00t_runtime_checkpoint_override_uses_model_path():
    spec = load_profile_spec(
        "configs/profiles.yaml", "official_gr00t", "g1_sonic_balance_simple"
    )
    config = simulation_config(
        spec,
        output_dir=Path("results/profiling/test"),
        profile_path=Path("profile.json"),
        profile_ref=Path("official_gr00t/2x-rpbzzz6/g1_sonic_balance_simple/instance_abc"),
        profile={
            "model_id": "official_gr00t",
            "gpu_class": "2x-rpbzzz6",
            "workload_id": "g1_sonic_balance_simple",
            "instance_id": "instance_abc",
            "source_run_id": "run_1",
            "env_fps": 50,
            "per_slot_summary": {"0": {}, "1": {}},
            "summary": {"mean_ms": 50.0, "std_ms": 4.0},
        },
        method="fixed",
        latency_seed=7,
        hf_commit="sha_123",
    )

    assert config["policy"]["model_path"] == spec["checkpoint_path"]
    assert "checkpoint_path" not in config["policy"]
    assert config["env"]["env_fps"] == 50
    assert config["env"]["obs_fps"] == 50


def test_fixed_workload_cadence_skips_policy_calibration(tmp_path, monkeypatch):
    spec = load_profile_spec(
        "configs/profiles.yaml", "official_gr00t", "g1_sonic_balance_simple"
    )
    monkeypatch.setattr(
        runner,
        "calibrate_env_fps_from_saturated_latency",
        lambda **_kwargs: pytest.fail("fixed cadence must skip policy calibration"),
    )

    calibration = runner._calibrate(
        spec,
        {"gpu_class": "2x-rpbzzz6", "instance_id": "instance_abc"},
        tmp_path,
    )

    assert calibration == {
        "selected_env_fps": 50,
        "selected_obs_fps": 50,
        "calibration_source": "fixed_workload_cadence",
    }
    assert json.loads((tmp_path / "calibration.json").read_text()) == calibration
    assert (tmp_path / "calibration.DONE").exists()
    assert not (tmp_path / "calibration.generated.yaml").exists()


def test_resume_run_id_cannot_escape_artifact_root():
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--model",
                "openvla",
                "--workload",
                "flappy",
                "--resume",
                "../run",
            ]
        )


def test_current_profile_replacement_keeps_only_new_bundle(tmp_path):
    current = tmp_path / "profiles" / "current"
    source = tmp_path / "candidate"
    current.mkdir(parents=True)
    source.mkdir()
    (current / "profile.json").write_text("old", encoding="utf-8")
    (source / "profile.json").write_text("new", encoding="utf-8")

    _replace_current(source, current)

    assert (current / "profile.json").read_text(encoding="utf-8") == "new"
    assert not current.with_name(".current.old").exists()


def test_profile_lock_serializes_current_writers(tmp_path):
    current = tmp_path / "profiles" / "current"
    entered = Event()

    def wait_for_lock():
        with _profile_lock(current):
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with _profile_lock(current):
            future = executor.submit(wait_for_lock)
            assert not entered.wait(0.05)
        future.result()

    assert entered.is_set()


def test_new_runs_are_retained_while_current_moves_forward(tmp_path, monkeypatch):
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps({"policy": {"checkpoint_path": "/models/openvla"}}),
        encoding="utf-8",
    )
    spec = {
        "artifact_root": str(tmp_path / "artifacts"),
        "model_id": "openvla",
        "workload_id": "flappy",
        "session_count": 2,
        "base_config": str(base_config),
        "checkpoint_path": "/models/openvla",
        "checkpoint_kind": "best",
        "gpu_models": {"rtx4090": {"aliases": ["NVIDIA RTX 4090"]}},
    }
    deployment = {
        "gpu_class": "2x-rtx4090",
        "instance_id": "instance_abc",
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    run_ids = iter(["run_1", "run_2"])
    monkeypatch.setattr(
        runner,
        "_collect_deployment",
        lambda _spec: (deployment, {"private": "hardware"}),
    )
    monkeypatch.setattr(
        runner,
        "_calibrate",
        lambda _spec, _deployment, _source: {"selected_env_fps": 25},
    )
    monkeypatch.setattr(runner, "_measure_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_new_run_id", lambda: next(run_ids))
    monkeypatch.setattr(
        runner,
        "public_hardware_projection",
        lambda *args, **kwargs: {"gpu_class": "2x-rtx4090"},
    )

    def build_bundle(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir()
        (output / "profile.json").write_text(
            json.dumps({"source_run_id": kwargs["source_run_id"]}),
            encoding="utf-8",
        )
        (output / "latency_distribution.json").write_text("{}", encoding="utf-8")
        (output / "latency_burst_model.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(runner, "build_profile_from_traces", build_bundle)

    first = runner.run_profile(spec, resume_run_id=None, publish=False)
    second = runner.run_profile(spec, resume_run_id=None, publish=False)

    coordinate = Path("openvla/2x-rtx4090/flappy/instance_abc")
    runs = Path(spec["artifact_root"]) / "runs" / coordinate
    current = Path(spec["artifact_root"]) / "profiles" / coordinate
    assert first["RUN_ID"] == "run_1"
    assert second["RUN_ID"] == "run_2"
    assert (runs / "run_1" / "DONE").exists()
    assert (runs / "run_2" / "DONE").exists()
    assert json.loads((current / "profile.json").read_text())["source_run_id"] == "run_2"


def test_simulation_uses_profile_slots_and_pinned_source(tmp_path):
    spec = load_profile_spec("configs/profiles.yaml", "openvla", "flappy")
    profile = {
        "model_id": "openvla",
        "gpu_class": "2x-rtx4090",
        "workload_id": "flappy",
        "instance_id": "instance_abc",
        "source_run_id": "run_1",
        "env_fps": 20,
        "per_slot_summary": {"0": {}, "1": {}},
        "summary": {"mean_ms": 50.0, "std_ms": 4.0},
    }

    config = simulation_config(
        spec,
        output_dir=tmp_path,
        profile_path=tmp_path / "profile.json",
        profile_ref=Path("openvla/2x-rtx4090/flappy/instance_abc"),
        profile=profile,
        method="temporal",
        latency_seed=7,
        hf_commit="sha_123",
    )

    assert config["executor"]["simulated_worker_capacity"] == 2
    assert config["policy"]["checkpoint_path"] == spec["checkpoint_path"]
    assert config["latency"]["source_run_id"] == "run_1"
    assert config["latency"]["hf_commit"] == "sha_123"


def test_publish_uses_one_commit_and_omits_private_hardware(tmp_path, monkeypatch):
    run = tmp_path / "run"
    profile = tmp_path / "profile"
    session = run / "sessions" / "0"
    session.mkdir(parents=True)
    (run / "profile").mkdir()
    profile.mkdir()
    (run / "hardware.json").write_text("private", encoding="utf-8")
    (run / "profile" / "profile.json").write_text("candidate", encoding="utf-8")
    (session / "actions.jsonl").write_text("{}\n", encoding="utf-8")
    for name in (
        "profile.json",
        "latency_distribution.json",
        "latency_burst_model.json",
        "hardware.json",
        "DONE",
    ):
        (profile / name).write_text("public", encoding="utf-8")

    client = SimpleNamespace(
        calls=[],
        create_commit=lambda **kwargs: (
            client.calls.append(kwargs) or SimpleNamespace(oid="commit_1")
        ),
    )

    class Add:
        def __init__(self, **kwargs):
            self.path_in_repo = kwargs["path_in_repo"]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(CommitOperationAdd=Add, HfApi=lambda: client),
    )

    commit = publish_run_and_profile(
        {"hf_repo": "org/profiles"},
        local_run_dir=run,
        local_profile_dir=profile,
        run_ref=Path("openvla/2x-rtx4090/flappy/instance_abc/run_1"),
        profile_ref=Path("openvla/2x-rtx4090/flappy/instance_abc"),
    )

    uploaded = {op.path_in_repo for op in client.calls[0]["operations"]}
    assert commit == "commit_1"
    assert len(client.calls) == 1
    assert not any(path.endswith("/hardware.json") and path.startswith("runs/") for path in uploaded)
    assert not any("/profile/" in path and path.startswith("runs/") for path in uploaded)
    assert "profiles/openvla/2x-rtx4090/flappy/instance_abc/hardware.json" in uploaded


def test_download_resolves_commit_before_fetching(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    profile = tmp_path / "profiles" / ref
    profile.mkdir(parents=True)
    for name in (
        "profile.json",
        "latency_distribution.json",
        "latency_burst_model.json",
        "hardware.json",
        "DONE",
    ):
        (profile / name).touch()
    calls = {}
    client = SimpleNamespace(repo_info=lambda **kwargs: SimpleNamespace(sha="sha_123"))

    def snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=lambda: client, snapshot_download=snapshot_download),
    )

    local, commit = download_profile(
        {"hf_repo": "org/profiles"},
        ref,
        hf_commit="main",
        local_dir=tmp_path / "downloads",
    )

    assert local == tmp_path / "downloads" / "sha_123"
    assert commit == "sha_123"
    assert calls["revision"] == "sha_123"


def test_download_materializes_hf_snapshot_symlinks(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    cache = tmp_path / "cache"
    source = cache / "profiles" / ref
    source.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for name in ("profile.json", "latency_distribution.json", "latency_burst_model.json", "hardware.json", "DONE"):
        target = blobs / name
        target.write_text(name, encoding="utf-8")
        (source / name).symlink_to(target)

    client = SimpleNamespace(repo_info=lambda **kwargs: SimpleNamespace(sha="sha_123"))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: client,
            snapshot_download=lambda **kwargs: str(cache),
        ),
    )

    local, commit = download_profile(
        {"hf_repo": "org/profiles"},
        ref,
        hf_commit="main",
        local_dir=tmp_path / "artifact_root" / "downloads" / "profiles" / ref,
    )

    assert commit == "sha_123"
    assert local == tmp_path / "artifact_root" / "downloads" / "profiles" / ref / "sha_123"
    assert all(not (local / name).is_symlink() for name in ("profile.json", "DONE"))
    assert (local / "profile.json").read_text(encoding="utf-8") == "profile.json"


def test_concurrent_download_materializes_a_commit_once(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    source = tmp_path / "cache" / "profiles" / ref
    source.mkdir(parents=True)
    for name in (
        "profile.json",
        "latency_distribution.json",
        "latency_burst_model.json",
        "hardware.json",
        "DONE",
    ):
        (source / name).write_text(name, encoding="utf-8")

    client = SimpleNamespace(repo_info=lambda **kwargs: SimpleNamespace(sha="sha_123"))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: client,
            snapshot_download=lambda **kwargs: str(tmp_path / "cache"),
        ),
    )
    copy_started = Event()
    release_copy = Event()
    copy_count = 0
    real_copyfile = shutil.copyfile

    def slow_copyfile(source_path, destination_path):
        nonlocal copy_count
        copy_count += 1
        if copy_count == 1:
            copy_started.set()
            release_copy.wait()
        return real_copyfile(source_path, destination_path)

    monkeypatch.setattr("latency_bench.cross_device.sync.shutil.copyfile", slow_copyfile)
    destination = tmp_path / "downloads"

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            download_profile,
            {"hf_repo": "org/profiles"},
            ref,
            hf_commit="main",
            local_dir=destination,
        )
        assert copy_started.wait(1.0)
        second = executor.submit(
            download_profile,
            {"hf_repo": "org/profiles"},
            ref,
            hf_commit="main",
            local_dir=destination,
        )
        try:
            assert not second.done()
        finally:
            release_copy.set()
        assert first.result() == second.result()

    assert copy_count == 5


def test_remote_profile_coordinates_must_match_request(tmp_path, monkeypatch):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "model_id": "openvla",
                "gpu_class": "1x-rtx4090",
                "workload_id": "flappy",
                "instance_id": "instance_other",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "latency_bench.cross_device.profile_resolver.download_profile",
        lambda *args, **kwargs: (tmp_path, "sha_123"),
    )

    with pytest.raises(ValueError, match="coordinates"):
        resolve_remote_profile(
            {
                "artifact_root": str(tmp_path / "artifacts"),
                "model_id": "openvla",
                "workload_id": "flappy",
            },
            gpu_class="2x-rtx4090",
            instance_id="instance_requested",
            hf_commit="main",
        )
