"""Long-lived profiling storage, config, and recovery contracts."""

import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from latency_bench.core.config import load_config
from latency_bench.profiling import runner
from latency_bench.profiling.artifacts import profile_dir, profile_ref, run_dir, run_ref, simulation_seed_dir
from latency_bench.profiling.profile_resolver import profile_refs_from_repo_files, resolve_remote_profile
from latency_bench.profiling.cli import main
from latency_bench.profiling.runner import _profile_lock, _replace_current
from latency_bench.profiling.runtime_configs import measure_config, simulation_config
from latency_bench.profiling.spec import resolve_profile_config
from latency_bench.profiling.sync import download_profile, publish_simulation_dirs


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


@pytest.mark.parametrize("artifact", ["measurement", "simulation"])
def test_publish_uses_one_commit_and_omits_local_configuration(tmp_path, monkeypatch, artifact):
    run = tmp_path / "run"
    profile = run / "profile"
    session = run / "sessions" / "0"
    session.mkdir(parents=True)
    profile.mkdir()
    (run / "hardware.json").write_text("private", encoding="utf-8")
    (run / "run_contract.json").write_text("private", encoding="utf-8")
    (session / "actions.jsonl").write_text("{}\n", encoding="utf-8")
    (session / "resolved_config.yaml").write_text("local paths", encoding="utf-8")
    (session / "generated.yaml").write_text("local paths", encoding="utf-8")
    (profile / "profile.json").write_text(
        json.dumps({"visualization_path": "latency_profile.png", "model_id": "openvla",
                    "gpu_class": "2x-rtx4090", "workload_id": "flappy",
                    "instance_id": "instance_abc", "source_run_id": "run_1"}),
        encoding="utf-8",
    )
    (profile / "latency_profile.png").write_bytes(b"png")
    for name in (
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

    (run / "DONE").touch()
    monkeypatch.setattr(runner, "collect_hardware_fingerprint", lambda: pytest.fail("publishing must not inspect this machine"))
    if artifact == "measurement":
        main(["publish", "--run-dir", str(run), "--repo", "org/profiles"])
    else:
        publish_simulation_dirs(
            "org/profiles", local_dirs=[session],
            remote_dirs=[Path("simulations/openvla/run_1/temporal/0")],
        )

    uploaded = {op.path_in_repo for op in client.calls[0]["operations"]}
    assert len(client.calls) == 1
    assert not any(path.endswith("/hardware.json") and path.startswith("runs/") for path in uploaded)
    assert not any(path.endswith("/run_contract.json") for path in uploaded)
    assert not any("/profile/" in path and path.startswith("runs/") for path in uploaded)
    assert not any(path.endswith(("/generated.yaml", "/resolved_config.yaml")) for path in uploaded)
    assert any(path.endswith("/actions.jsonl") for path in uploaded)
    if artifact == "simulation":
        return
    assert "profiles/openvla/2x-rtx4090/flappy/instance_abc/hardware.json" in uploaded
    assert "profiles/openvla/2x-rtx4090/flappy/instance_abc/latency_profile.png" in uploaded
    profile_uploads = [
        operation.path_in_repo
        for operation in client.calls[0]["operations"]
        if operation.path_in_repo.startswith("profiles/")
    ]
    assert profile_uploads[-2:] == [
        "profiles/openvla/2x-rtx4090/flappy/instance_abc/latency_profile.png",
        "profiles/openvla/2x-rtx4090/flappy/instance_abc/DONE",
    ]


def test_download_fetches_current_profile(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    snapshots = [tmp_path / "cache" / "snapshots" / (commit * 40) for commit in "ab"]
    for snapshot, run_id in zip(snapshots, ["run_A", "run_B"]):
        profile = snapshot / "profiles" / ref
        profile.mkdir(parents=True)
        for name in (
            "latency_distribution.json",
            "latency_burst_model.json",
            "hardware.json",
            "DONE",
        ):
            (profile / name).write_text(run_id, encoding="utf-8")
        (profile / "profile.json").write_text(
            json.dumps({"source_run_id": run_id}), encoding="utf-8"
        )
    destination = tmp_path / "downloads"
    shutil.copytree(snapshots[0] / "profiles" / ref, destination)
    current_snapshots = iter(snapshots)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(next(current_snapshots))

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download),
    )

    first = download_profile(
        "org/profiles",
        ref,
        local_dir=destination,
    )
    second = download_profile(
        "org/profiles",
        ref,
        local_dir=destination,
    )

    assert first == destination / snapshots[0].name
    assert second == destination / snapshots[1].name
    assert json.loads((first / "profile.json").read_text())["source_run_id"] == "run_A"
    assert json.loads((second / "profile.json").read_text())["source_run_id"] == "run_B"
    assert (first / "latency_distribution.json").read_text() == "run_A"
    assert (second / "latency_distribution.json").read_text() == "run_B"
    assert all("revision" not in call for call in calls)


def test_download_requires_and_repairs_declared_visualization(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    cache = tmp_path / "cache" / "snapshots" / ("a" * 40)
    source = cache / "profiles" / ref
    source.mkdir(parents=True)
    (source / "profile.json").write_text(
        json.dumps({"visualization_path": "latency_profile.png"}),
        encoding="utf-8",
    )
    for name in (
        "latency_distribution.json",
        "latency_burst_model.json",
        "hardware.json",
        "DONE",
    ):
        (source / name).write_text(name, encoding="utf-8")

    local_dir = tmp_path / "downloads"
    shutil.copytree(source, local_dir)
    newer_cache = cache.with_name("b" * 40)
    newer_source = newer_cache / "profiles" / ref
    newer_source.mkdir(parents=True)
    (newer_source / "latency_profile.png").write_bytes(b"newer png")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if "revision" not in kwargs and kwargs["allow_patterns"] == [
            f"profiles/{ref.as_posix()}/latency_profile.png"
        ]:
            return str(newer_cache)
        return str(cache)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    with pytest.raises(FileNotFoundError):
        download_profile(
            "org/profiles",
            ref,
            local_dir=local_dir,
        )
    assert not (local_dir / cache.name).exists()
    assert list(local_dir.glob(f".{cache.name}.*")) == []
    (source / "latency_profile.png").write_bytes(b"png")

    local = download_profile(
        "org/profiles",
        ref,
        local_dir=local_dir,
    )

    assert local == local_dir / cache.name
    assert (local / "latency_profile.png").read_bytes() == b"png"
    assert [call["revision"] for call in calls if "revision" in call] == [cache.name, cache.name]


def test_download_materializes_hf_snapshot_symlinks(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    cache = tmp_path / "cache" / "snapshots" / ("a" * 40)
    source = cache / "profiles" / ref
    source.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for name in ("profile.json", "latency_distribution.json", "latency_burst_model.json", "hardware.json", "DONE"):
        target = blobs / name
        target.write_text("{}" if name == "profile.json" else name, encoding="utf-8")
        (source / name).symlink_to(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: str(cache),
        ),
    )

    local = download_profile(
        "org/profiles",
        ref,
        local_dir=tmp_path / "artifact_root" / "downloads" / "profiles" / ref,
    )

    assert local == tmp_path / "artifact_root" / "downloads" / "profiles" / ref / cache.name
    assert all(not (local / name).is_symlink() for name in ("profile.json", "DONE"))
    assert (local / "profile.json").read_text(encoding="utf-8") == "{}"


def test_concurrent_download_materializes_a_profile_once(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    cache = tmp_path / "cache" / "snapshots" / ("a" * 40)
    source = cache / "profiles" / ref
    source.mkdir(parents=True)
    for name in (
        "profile.json",
        "latency_distribution.json",
        "latency_burst_model.json",
        "hardware.json",
        "DONE",
    ):
        (source / name).write_text(
            "{}" if name == "profile.json" else name,
            encoding="utf-8",
        )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: str(cache),
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

    monkeypatch.setattr("latency_bench.profiling.sync.shutil.copyfile", slow_copyfile)
    destination = tmp_path / "downloads"

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            download_profile,
            "org/profiles",
            ref,
            local_dir=destination,
        )
        assert copy_started.wait(1.0)
        second = executor.submit(
            download_profile,
            "org/profiles",
            ref,
            local_dir=destination,
        )
        try:
            assert not second.done()
        finally:
            release_copy.set()
        assert first.result() == second.result()

    assert copy_count == 5


def test_interrupted_download_does_not_install_partial_revision(tmp_path, monkeypatch):
    ref = Path("openvla/2x-rtx4090/flappy/instance_abc")
    cache = tmp_path / "cache" / "snapshots" / ("a" * 40)
    source = cache / "profiles" / ref
    source.mkdir(parents=True)
    for name in (
        "profile.json",
        "latency_distribution.json",
        "latency_burst_model.json",
        "hardware.json",
        "DONE",
    ):
        (source / name).write_text("{}", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: str(cache)),
    )
    copyfile = shutil.copyfile

    def interrupted_copyfile(source_path, destination_path):
        if source_path.name == "latency_distribution.json":
            destination_path.write_text("{TRUNCATED", encoding="utf-8")
            raise OSError("interrupted copy")
        return copyfile(source_path, destination_path)

    destination = tmp_path / "downloads"
    monkeypatch.setattr("latency_bench.profiling.sync.shutil.copyfile", interrupted_copyfile)
    with pytest.raises(OSError, match="interrupted copy"):
        download_profile("org/profiles", ref, local_dir=destination)
    assert sorted(path.name for path in destination.iterdir()) == [".lock"]

    monkeypatch.setattr("latency_bench.profiling.sync.shutil.copyfile", copyfile)
    local = download_profile("org/profiles", ref, local_dir=destination)
    assert local == destination / cache.name
    assert json.loads((local / "latency_distribution.json").read_text()) == {}
    assert (local / "DONE").exists()


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
        "latency_bench.profiling.profile_resolver.download_profile",
        lambda *args, **kwargs: tmp_path,
    )

    with pytest.raises(ValueError, match="coordinates"):
        resolve_remote_profile(
            "org/profiles", "openvla/2x-rtx4090/flappy/instance_requested", output_dir=tmp_path,
        )


def test_measurement_preserves_user_deployment_and_clock(profile_config, tmp_path):
    config = resolve_profile_config(profile_config)
    config["executor"].update(inference_devices=["cuda:2", "cuda:2"], inference_batch_size=1)
    config["policy"]["worker_python_executable"] = "/user/python"
    original = deepcopy(config)
    measured = measure_config(config, {"gpu_class": "1x-custom", "instance_id": "instance_a"}, tmp_path / "session")
    for section in ("executor", "policy", "env", "evaluation"):
        assert measured[section] == original[section]
    assert config == original
    assert measured["logging"]["video"]["enabled"] is False
    assert measured["logging"]["save_latency_records"] is True


@pytest.mark.parametrize("field,value", [("repeats", 0), ("model_id", "../escape")])
def test_invalid_profile_config_is_rejected(profile_config, field, value):
    profile_config["profiling"][field] = value
    with pytest.raises(ValueError):
        resolve_profile_config(profile_config)


def test_profiling_rejects_ambiguous_sweep(profile_config):
    profile_config["evaluation"]["suites"] = [{"name": "fixed_0", "latency": {"method": "zero"}}]
    with pytest.raises(ValueError, match="sweeps"):
        resolve_profile_config(profile_config)


@pytest.mark.parametrize("method", ["temporal", "fixed", "normal"])
def test_local_replay_preserves_target_clock_and_uses_profile_capacity(profile_config, tmp_path, method):
    profile = {"model_id": "measured", "workload_id": "source", "gpu_class": "2x-gpu",
               "instance_id": "instance_a", "source_run_id": "run_a", "env_fps": 7,
               "per_slot_summary": [{}, {}], "summary": {"mean_ms": 60, "std_ms": 10}}
    replay = simulation_config(profile_config, output_dir=tmp_path / "replay", profile_path=tmp_path / "profile.json",
                               profile=profile, method=method, latency_seed=42)
    assert replay["env"] == profile_config["env"]
    assert replay["executor"]["simulated_worker_capacity"] == 2
    assert replay["executor"]["inference_devices"] == ["cpu"]
    assert replay["latency"]["seed"] == 42
    if method == "fixed":
        assert replay["latency"]["fixed_latency_ms"] == 60
    if method == "normal":
        assert replay["latency"]["std_latency_ms"] == 10


def test_cli_reads_the_same_run_config(profile_config, tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(profile_config))
    captured = {}
    monkeypatch.setattr("latency_bench.profiling.cli.run_profile", lambda config, **kwargs:
                        captured.update(config=config, kwargs=kwargs) or {"PROFILE_PATH": "/local/profile.json"})
    main(["run", "--config", str(path)])
    assert captured == {"config": profile_config, "kwargs": {"resume_run_id": None, "publish": False}}
    assert "PROFILE_PATH=/local/profile.json" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["run", "--config", str(path), "--resume", "../escape"])


@pytest.mark.parametrize("calibration_requested", [False, True])
def test_completed_sessions_resume_and_fit_without_remeasurement(
    profile_config, tmp_path, monkeypatch, calibration_requested,
):
    profile_config["experiment"].pop("seed")
    monkeypatch.delenv("LATENCY_BENCH_ENV_BACKEND_START_METHOD", raising=False)
    hardware = {"cpu": {"model_name": "test", "logical_cores": 4}, "system": {"hostname": "host"},
                "gpus": [], "gpu": {}, "torch": {}, "memory": {}}
    monkeypatch.setattr(runner, "collect_hardware_fingerprint", lambda: hardware)
    monkeypatch.setattr(runner, "_new_run_id", lambda: "run_1")
    calls = []
    fail_second = True
    calibration_calls = []
    if calibration_requested:
        profile_config["profiling"]["calibration"] = {
            "target_latency_frames": 2, "worker_utilization_cap": 0.8,
        }

    def calibrate(*, config_path, output, **options):
        calibration_calls.append(config_path)
        result = {"selected_env_fps": 100, "selected_obs_fps": 20}
        Path(output).write_text(json.dumps(result))
        return result

    monkeypatch.setattr(runner, "calibrate_env_fps_from_saturated_latency", calibrate)

    def evaluate(config, generated, output):
        nonlocal fail_second
        calls.append(output.name)
        if output.name == "1" and fail_second:
            fail_second = False
            raise RuntimeError("interrupted session")
        rows = [{"obs_id": i, "episode_id": i // 20, "obs_env_step": i % 20,
                 "submitted": i % 7 != 0, "worker_id": 0,
                 "observation_to_action_latency_ms": 20.0 + i % 4,
                 "worker_service_time_ms": 18.0 + i % 4} for i in range(40)]
        (output / "observation_attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    monkeypatch.setattr(runner, "_run_evaluation", evaluate)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run_profile(profile_config)
    source = next(Path(profile_config["logging"]["output_dir"]).rglob("run_contract.json")).parent
    assert (source / "sessions/0/DONE").exists()
    assert not (source / "DONE").exists()
    profile_config["profiling"]["publish_repo"] = "org/new-publication-target"
    result = runner.run_profile(profile_config, resume_run_id="run_1")
    assert calls == ["0", "1", "1"]
    assert result["PUBLICATION_STATUS"] == "local"
    profile = json.loads(Path(result["PROFILE_PATH"]).read_text())
    assert profile["n_observation_attempts"] == 80
    assert profile["n_capacity_drops"] == 12
    assert profile["provenance"]["session_ids"] == [0, 1]
    assert profile["provenance"]["measurement"]["seed"] == 0
    assert profile["env_fps"] == (100 if calibration_requested else 50)
    assert len(calibration_calls) == int(calibration_requested)
    monkeypatch.setenv("LATENCY_BENCH_ENV_BACKEND_START_METHOD", "fork")
    with pytest.raises(ValueError, match="run contract does not match"):
        runner.run_profile(profile_config, resume_run_id="run_1")
    monkeypatch.delenv("LATENCY_BENCH_ENV_BACKEND_START_METHOD")
    profile_config["evaluation"]["episodes"] = 3
    with pytest.raises(ValueError, match="run contract does not match"):
        runner.run_profile(profile_config, resume_run_id="run_1")
    assert calls == ["0", "1", "1"]
    assert (source / "calibration.json").exists() == calibration_requested
    assert (source / "calibration.DONE").exists() == calibration_requested
    original_profile = profile
    shutil.rmtree(source / "profile")
    monkeypatch.setattr(runner, "collect_hardware_fingerprint", lambda: pytest.fail("fit must use recorded hardware"))
    refit = runner.fit_profile(source)
    assert json.loads(Path(refit["PROFILE_PATH"]).read_text()) == original_profile
    assert calls == ["0", "1", "1"]
