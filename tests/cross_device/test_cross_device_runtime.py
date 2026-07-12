from __future__ import annotations

import sys
import types
import shutil
from pathlib import Path

import pytest
import yaml

from latency_bench.cross_device import protocol, runner, sync
from latency_bench.cross_device.spec import load_workload
from latency_bench.utils.io import read_json, write_json


def _workload(tmp_path: Path) -> dict:
    workload = load_workload("configs/experiments/cross_device.yaml")
    workload["artifact_root"] = str(tmp_path / "artifacts")
    return workload


def _install_runtime_fakes(monkeypatch, calls: dict) -> None:
    monkeypatch.setenv("CROSS_DEVICE_INSTANCE_ID", "instance_a")
    monkeypatch.setattr(
        runner,
        "collect_hardware_fingerprint",
        lambda: {"gpu": {"uuid": "GPU-a100", "name": "A100"}},
    )

    def fake_calibration(**kwargs):
        calls["calibrations"].append(kwargs)
        calibration = {"selected_env_fps": 16.0, "selected_obs_fps": 16.0}
        write_json(kwargs["output"], calibration)
        return calibration

    def fake_eval(_workload, _config_path, output_dir):
        calls["evaluations"].append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "e2e_latencies.jsonl",
            "infer_latencies.jsonl",
            "episode_metrics.jsonl",
            "actions.jsonl",
            "steps.jsonl",
        ):
            (output_dir / name).write_text("", encoding="utf-8")

    monkeypatch.setattr(runner, "calibrate_env_fps_from_saturated_latency", fake_calibration)
    monkeypatch.setattr(runner, "_run_evaluation", fake_eval)
    monkeypatch.setattr(runner, "download_calibration", lambda *_args: False)


def _raw_dir(workload: dict) -> Path:
    return (
        Path(workload["artifact_root"])
        / "raw/a100/instance_a/openvla-flappy"
    )


def test_measure_appends_sessions_and_reuses_calibration(tmp_path, monkeypatch):
    workload = _workload(tmp_path)
    calls = {"calibrations": [], "evaluations": []}
    _install_runtime_fakes(monkeypatch, calls)

    runner.run_measure(workload, "a100", [0])
    runner.run_measure(workload, "a100", [1, 2])

    raw_dir = _raw_dir(workload)
    assert len(calls["calibrations"]) == 1
    assert [path.name for path in calls["evaluations"]] == ["0", "1", "2"]
    assert (raw_dir / "calibration.DONE").is_file()
    assert not (raw_dir / "DONE").exists()
    for session_id in range(3):
        session_dir = raw_dir / "sessions" / str(session_id)
        assert (session_dir / "DONE").is_file()
        assert read_json(session_dir / "calibration.json")["selected_env_fps"] == 16.0
        assert read_json(session_dir / "hardware.json")["gpu"]["name"] == "A100"


def test_measure_resume_skips_done_and_replaces_partial_leaf(tmp_path, monkeypatch):
    workload = _workload(tmp_path)
    calls = {"calibrations": [], "evaluations": []}
    _install_runtime_fakes(monkeypatch, calls)
    runner.run_measure(workload, "a100", [0])
    partial_dir = _raw_dir(workload) / "sessions/1"
    partial_dir.mkdir(parents=True)
    (partial_dir / "partial.txt").write_text("keep", encoding="utf-8")

    runner.run_measure(workload, "a100", [0, 1], resume=True)

    assert [path.name for path in calls["evaluations"]] == ["0", "1"]
    assert not (partial_dir / "partial.txt").exists()
    with pytest.raises(FileExistsError):
        runner.run_measure(workload, "a100", [0])


def test_measure_reuses_remote_calibration_when_local_copy_is_missing(tmp_path, monkeypatch):
    workload = _workload(tmp_path)
    calls = {"calibrations": [], "evaluations": []}
    _install_runtime_fakes(monkeypatch, calls)

    def fake_download(_workload, _target, raw_dir):
        write_json(
            raw_dir / "calibration.json",
            {"selected_env_fps": 20.0, "selected_obs_fps": 20.0},
        )
        write_json(raw_dir / "hardware.json", {"gpu": {"uuid": "GPU-remote"}})
        (raw_dir / "calibration.generated.yaml").write_text("{}", encoding="utf-8")
        (raw_dir / "calibration.DONE").touch()
        return True

    monkeypatch.setattr(runner, "download_calibration", fake_download)

    runner.run_measure(workload, "a100", [0])

    assert calls["calibrations"] == []
    session_dir = _raw_dir(workload) / "sessions/0"
    assert read_json(session_dir / "calibration.json")["selected_env_fps"] == 20.0
    assert read_json(session_dir / "hardware.json")["gpu"]["uuid"] == "GPU-remote"


def test_fit_creates_independent_immutable_revisions(tmp_path, monkeypatch):
    workload = _workload(tmp_path)
    calls = {"calibrations": [], "evaluations": [], "profiles": []}
    _install_runtime_fakes(monkeypatch, calls)
    runner.run_measure(workload, "a100", [0, 1, 2])

    def fake_build(**kwargs):
        calls["profiles"].append(kwargs)
        write_json(
            Path(kwargs["output_dir"]) / "profile.json",
            {
                "profile_id": kwargs["profile_id"],
                "env_fps": kwargs["env_fps"],
                "inference_worker_count": 2,
                **kwargs["metadata"],
            },
        )

    monkeypatch.setattr(runner, "build_profile_from_traces", fake_build)
    target = Path("a100/instance_a/openvla-flappy")
    partial_profile = (
        Path(workload["artifact_root"])
        / "profiles/a100/instance_a/openvla-flappy/fit-0"
    )
    partial_profile.mkdir(parents=True)
    (partial_profile / "partial.txt").write_text("discard", encoding="utf-8")

    first = runner.run_fit(workload, target, "fit-0", [0])
    second = runner.run_fit(workload, target, "fit-0-2", [0, 2])

    assert first == Path("a100/instance_a/openvla-flappy/fit-0")
    assert second == Path("a100/instance_a/openvla-flappy/fit-0-2")
    assert [
        [session["measure_session_id"] for session in call["sessions"]]
        for call in calls["profiles"]
    ] == [[0], [0, 2]]
    first_dir = Path(workload["artifact_root"]) / "profiles" / first
    assert not (first_dir / "partial.txt").exists()
    assert read_json(first_dir / "profile.json")["fit_session_ids"] == [0]
    assert read_json(first_dir / "hardware.json")["gpu"]["uuid"] == "GPU-a100"
    with pytest.raises(FileExistsError):
        runner.run_fit(workload, target, "fit-0", [0])
    shutil.rmtree(_raw_dir(workload))
    assert runner.run_fit(workload, target, "fit-0", [0], resume=True) == first


def test_simulation_appends_explicit_latency_seeds(tmp_path, monkeypatch):
    workload = _workload(tmp_path)
    calls = {"calibrations": [], "evaluations": []}
    _install_runtime_fakes(monkeypatch, calls)
    canonical_ref = Path("a100/instance_a/openvla-flappy/fit-0")
    downloaded = tmp_path / "downloaded-profile"
    write_json(
        downloaded / "profile.json",
        {
            "profile_id": "L_A100",
            "env_fps": 16.0,
            "inference_worker_count": 2,
            "device_id": "a100",
            "instance_id": "instance_a",
            "profile_revision": "fit-0",
        },
    )
    (downloaded / "DONE").touch()
    monkeypatch.setattr(runner, "download_profile", lambda *_args: downloaded)
    partial_seed = (
        Path(workload["artifact_root"])
        / "sim"
        / canonical_ref
        / "temporal/0"
    )
    partial_seed.mkdir(parents=True)
    (partial_seed / "partial.txt").write_text("discard", encoding="utf-8")

    runner.run_simulation(workload, canonical_ref, "temporal", [0, 2])
    runner.run_simulation(workload, canonical_ref, "temporal", [1, 7])

    method_dir = Path(workload["artifact_root"]) / "sim" / canonical_ref / "temporal"
    assert [path.name for path in calls["evaluations"]] == ["0", "2", "1", "7"]
    assert not (partial_seed / "partial.txt").exists()
    assert not (method_dir / "DONE").exists()
    generated = yaml.safe_load((method_dir / "7/generated.yaml").read_text(encoding="utf-8"))
    assert generated["latency"]["profile_ref"] == canonical_ref.as_posix()
    assert generated["latency"]["seed"] == 7
    assert "simulation_host" not in generated["latency"]
    with pytest.raises(FileExistsError):
        runner.run_simulation(workload, canonical_ref, "temporal", [0])
    monkeypatch.setattr(
        runner,
        "download_profile",
        lambda *_args: pytest.fail("completed resume must not download profile"),
    )
    runner.run_simulation(workload, canonical_ref, "temporal", [0], resume=True)


class _FakeHfApi:
    existing: set[str] = set()
    calls: list[tuple] = []

    def file_exists(self, *, filename, **_kwargs):
        return filename in self.existing

    def upload_folder(self, **kwargs):
        self.calls.append(("folder", kwargs["path_in_repo"]))

    def upload_file(self, **kwargs):
        self.calls.append(("file", kwargs["path_in_repo"]))


def test_publish_is_append_only_and_writes_done_last(tmp_path, monkeypatch):
    workload = _workload(tmp_path)
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    (leaf / "payload.json").write_text("{}", encoding="utf-8")
    (leaf / "DONE").touch()
    _FakeHfApi.existing = set()
    _FakeHfApi.calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_FakeHfApi))

    sync.publish_directory_leaf(workload, leaf, Path("profiles/target/profile"))

    assert _FakeHfApi.calls == [
        ("folder", "profiles/target/profile"),
        ("file", "profiles/target/profile/DONE"),
    ]
    _FakeHfApi.existing = {"profiles/target/profile/DONE"}
    with pytest.raises(FileExistsError):
        sync.publish_directory_leaf(workload, leaf, Path("profiles/target/profile"))
    for path in leaf.iterdir():
        path.unlink()
    leaf.rmdir()
    sync.publish_directory_leaf(
        workload,
        leaf,
        Path("profiles/target/profile"),
        resume=True,
    )


def test_calibration_publish_writes_snapshot_before_done(tmp_path, monkeypatch):
    workload = _workload(tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in ("calibration.generated.yaml", "calibration.json", "hardware.json"):
        (raw_dir / name).write_text("{}", encoding="utf-8")
    (raw_dir / "calibration.DONE").touch()
    _FakeHfApi.existing = set()
    _FakeHfApi.calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_FakeHfApi))

    sync.publish_calibration(workload, raw_dir, Path("raw/a100/instance/workload"))

    assert _FakeHfApi.calls == [
        ("file", "raw/a100/instance/workload/calibration.generated.yaml"),
        ("file", "raw/a100/instance/workload/calibration.json"),
        ("file", "raw/a100/instance/workload/hardware.json"),
        ("file", "raw/a100/instance/workload/calibration.DONE"),
    ]


def test_analysis_download_requests_only_selected_leaves(tmp_path, monkeypatch):
    workload = _workload(tmp_path)
    profile_ref = Path("a100/instance/workload/revision")
    profile_dir = tmp_path / "profile"
    write_json(profile_dir / "profile.json", {"fit_session_ids": [0, 2]})
    calls = []
    monkeypatch.setattr(sync, "download_profile", lambda *_args: profile_dir)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            snapshot_download=lambda **kwargs: calls.append(kwargs) or str(tmp_path)
        ),
    )

    root = sync.download_analysis_selection(
        workload,
        [profile_ref],
        [5],
        ["temporal", "iid"],
        [1, 3],
    )

    assert root == tmp_path
    assert set(calls[0]["allow_patterns"]) == {
        "profiles/a100/instance/workload/revision/**",
        "raw/a100/instance/workload/sessions/0/**",
        "raw/a100/instance/workload/sessions/2/**",
        "raw/a100/instance/workload/sessions/5/**",
        "sim/a100/instance/workload/revision/temporal/1/**",
        "sim/a100/instance/workload/revision/temporal/3/**",
        "sim/a100/instance/workload/revision/iid/1/**",
        "sim/a100/instance/workload/revision/iid/3/**",
    }


def test_protocol_dispatches_explicit_simulation_selection(monkeypatch):
    workload = {"workload": {"id": "openvla-flappy"}}
    calls = []
    monkeypatch.setattr(protocol, "load_workload", lambda _path: workload)
    monkeypatch.setattr(
        protocol,
        "run_simulation",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    protocol.main(
        [
            "sim",
            "--profile-ref",
            "a100/instance_a/openvla-flappy/fit-0",
            "--method",
            "iid",
            "--latency-seeds",
            "0",
            "2",
            "7",
            "--resume",
        ]
    )

    assert calls == [
        (
            (
                workload,
                Path("a100/instance_a/openvla-flappy/fit-0"),
                "iid",
                [0, 2, 7],
            ),
            {"resume": True},
        )
    ]


def test_protocol_aggregate_dispatches_explicit_artifact_selection(
    tmp_path,
    monkeypatch,
):
    workload = {"artifact_root": str(tmp_path), "workload": {"id": "workload"}}
    profile_refs = [Path("a100/instance/workload/revision")]
    calls = []
    monkeypatch.setattr(protocol, "load_workload", lambda _path: workload)
    monkeypatch.setattr(
        protocol,
        "download_analysis_selection",
        lambda *args: calls.append(("download", args)) or tmp_path / "remote",
    )
    monkeypatch.setattr(
        protocol,
        "write_cross_device_outputs",
        lambda *args: calls.append(("write", args)) or {},
    )

    protocol.main(
        [
            "aggregate",
            "--profile-ref",
            str(profile_refs[0]),
            "--real-sessions",
            "5",
            "6",
            "--methods",
            "temporal",
            "iid",
            "--latency-seeds",
            "0",
            "2",
        ]
    )

    selection = (workload, profile_refs, [5, 6], ["temporal", "iid"], [0, 2])
    assert calls == [
        ("download", selection),
        (
            "write",
            (
                workload,
                tmp_path / "remote",
                tmp_path / "analysis",
                profile_refs,
                [5, 6],
                ["temporal", "iid"],
                [0, 2],
            ),
        ),
    ]
