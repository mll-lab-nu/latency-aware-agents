from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

from latency_bench.cross_device import protocol, runner, sync
from latency_bench.cross_device.spec import load_study
from latency_bench.utils.io import read_json, write_json


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _study(tmp_path: Path, name: str) -> dict:
    study = load_study(Path("configs/experiments") / name)
    study["output_root"] = str(tmp_path / "results")
    return study


def _install_measure_fakes(monkeypatch, calls: dict) -> None:
    monkeypatch.setenv("CROSS_DEVICE_INSTANCE_ID", "instance_a")
    monkeypatch.setattr(
        runner,
        "collect_hardware_fingerprint",
        lambda: {"gpu": {"uuid": "GPU-a100", "name": "A100"}},
    )

    def fake_calibration(**kwargs):
        calls["calibrations"].append(kwargs)
        payload = {"selected_env_fps": 16.0, "selected_obs_fps": 16.0}
        write_json(kwargs["output"], payload)
        return payload

    def fake_eval(_study, _config_path, output_dir):
        calls["sessions"].append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "e2e_latencies.jsonl",
            "infer_latencies.jsonl",
            "episode_metrics.jsonl",
            "actions.jsonl",
            "steps.jsonl",
        ):
            (output_dir / name).write_text("", encoding="utf-8")

    def fake_build(**kwargs):
        calls["profiles"].append(kwargs)
        profile_dir = Path(kwargs["output_dir"])
        write_json(
            profile_dir / "profile.json",
            {
                "env_fps": kwargs["env_fps"],
                "inference_worker_count": 2,
                **kwargs["metadata"],
            },
        )
        return read_json(profile_dir / "profile.json")

    monkeypatch.setattr(runner, "calibrate_env_fps_from_saturated_latency", fake_calibration)
    monkeypatch.setattr(runner, "_run_evaluation", fake_eval)
    monkeypatch.setattr(runner, "build_profile_from_traces", fake_build)


@pytest.mark.parametrize(
    ("config_name", "session_ids", "fit_ids"),
    [
        ("cross_device_1real_3sim.yaml", [0], [0]),
        ("cross_device_10real_20sim.yaml", list(range(10)), list(range(5))),
    ],
)
def test_measure_runs_each_real_session_once(
    tmp_path,
    monkeypatch,
    config_name,
    session_ids,
    fit_ids,
):
    study = _study(tmp_path, config_name)
    calls = {"calibrations": [], "sessions": [], "profiles": []}
    _install_measure_fakes(monkeypatch, calls)

    runner.run_measure(study, "a100")

    measure_dir = Path(study["output_root"]) / "measure/a100/instance_a"
    assert [path.name for path in calls["sessions"]] == [str(value) for value in session_ids]
    assert len(calls["calibrations"]) == 1
    assert [item["measure_session_id"] for item in calls["profiles"][0]["sessions"]] == fit_ids
    assert calls["profiles"][0]["metadata"]["evaluation_session_ids"] == study["sessions"]["evaluation"]
    assert (measure_dir / "DONE").is_file()
    assert read_json(measure_dir / "profile/profile.json")["hardware"]["gpu"]["name"] == "A100"


def test_failed_measure_does_not_write_done(tmp_path, monkeypatch):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    calls = {"calibrations": [], "sessions": [], "profiles": []}
    _install_measure_fakes(monkeypatch, calls)
    monkeypatch.setattr(
        runner,
        "_run_evaluation",
        lambda *_args: _fail("evaluation failed"),
    )

    with pytest.raises(RuntimeError, match="evaluation failed"):
        runner.run_measure(study, "a100")

    assert not (Path(study["output_root"]) / "measure/a100/instance_a/DONE").exists()


def test_measure_resume_reuses_calibration_and_completed_sessions(tmp_path, monkeypatch):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    study["sessions"] = {"fit": [0], "evaluation": [1]}
    calls = {"calibrations": [], "sessions": [], "profiles": []}
    _install_measure_fakes(monkeypatch, calls)
    measure_dir = Path(study["output_root"]) / "measure/a100/instance_a"
    write_json(
        measure_dir / "calibration.json",
        {"selected_env_fps": 16.0, "selected_obs_fps": 16.0},
    )
    (measure_dir / "calibration.DONE").touch()
    completed_session = measure_dir / "sessions/0"
    completed_session.mkdir(parents=True)
    for name in (
        "e2e_latencies.jsonl",
        "infer_latencies.jsonl",
        "episode_metrics.jsonl",
        "actions.jsonl",
        "steps.jsonl",
    ):
        (completed_session / name).write_text("", encoding="utf-8")
    (completed_session / "DONE").touch()
    incomplete_session = measure_dir / "sessions/1"
    incomplete_session.mkdir(parents=True)
    (incomplete_session / "stale").touch()
    incomplete_profile = measure_dir / "profile"
    incomplete_profile.mkdir()
    (incomplete_profile / "stale").touch()

    runner.run_measure(study, "a100", resume=True)

    assert calls["calibrations"] == []
    assert [path.name for path in calls["sessions"]] == ["1"]
    assert [item["measure_session_id"] for item in calls["profiles"][0]["sessions"]] == [0]
    assert not (incomplete_session / "stale").exists()
    assert not (incomplete_profile / "stale").exists()
    assert (measure_dir / "DONE").is_file()


def test_measure_resume_reuses_completed_profile_and_group(tmp_path, monkeypatch):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    calls = {"calibrations": [], "sessions": [], "profiles": []}
    _install_measure_fakes(monkeypatch, calls)
    measure_dir = Path(study["output_root"]) / "measure/a100/instance_a"
    write_json(
        measure_dir / "calibration.json",
        {"selected_env_fps": 16.0, "selected_obs_fps": 16.0},
    )
    (measure_dir / "calibration.DONE").touch()
    session_dir = measure_dir / "sessions/0"
    session_dir.mkdir(parents=True)
    (session_dir / "DONE").touch()
    profile_dir = measure_dir / "profile"
    write_json(profile_dir / "profile.json", {"env_fps": 16.0})
    (profile_dir / "DONE").touch()

    runner.run_measure(study, "a100", resume=True)
    runner.run_measure(study, "a100", resume=True)

    assert calls == {"calibrations": [], "sessions": [], "profiles": []}
    assert (measure_dir / "DONE").is_file()


def _write_measurement(root: Path, device_id: str, instance_id: str) -> None:
    measure_dir = root / "measure" / device_id / instance_id
    write_json(
        measure_dir / "profile/profile.json",
        {"env_fps": 16.0, "inference_worker_count": 2},
    )
    (measure_dir / "DONE").touch()


def test_simulation_runs_every_profile_instance_without_host_identity(tmp_path, monkeypatch):
    study = _study(tmp_path, "cross_device_10real_20sim.yaml")
    study["latency_seeds"] = [0, 1]
    measurements = tmp_path / "download"
    for device in study["devices"]:
        _write_measurement(measurements, device["id"], "instance_a")
    _write_measurement(measurements, "a100", "instance_b")
    configs = []

    monkeypatch.setattr(runner, "download_measurements", lambda _study: measurements)

    def fake_eval(_study, config_path, _output_dir):
        configs.append(yaml.safe_load(config_path.read_text(encoding="utf-8")))

    monkeypatch.setattr(runner, "_run_evaluation", fake_eval)

    runner.run_simulation(study, "temporal")

    assert len(configs) == 10
    assert {config["latency"]["method"] for config in configs} == {"temporal"}
    assert {config["latency"]["seed"] for config in configs} == {0, 1}
    assert {config["latency"]["instance_id"] for config in configs} == {
        "instance_a",
        "instance_b",
    }
    assert all("replay_host" not in config["latency"] for config in configs)
    assert all(config["executor"]["simulated_worker_capacity"] == 2 for config in configs)
    assert all(config["env"]["env_fps"] == 16.0 for config in configs)
    assert all(config["latency"]["profile_path"].endswith("profile/profile.json") for config in configs)
    sim_root = Path(study["output_root"]) / "sim"
    assert (sim_root / "a100/instance_a/temporal/DONE").is_file()
    assert (sim_root / "a100/instance_b/temporal/DONE").is_file()
    assert (sim_root / "rtx5090/instance_a/temporal/DONE").is_file()


def test_simulation_runs_the_completed_profiles_available_to_the_host(tmp_path, monkeypatch):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    study["latency_seeds"] = [0]
    measurements = tmp_path / "download"
    _write_measurement(measurements, "a100", "instance_a")
    calls = []
    monkeypatch.setattr(runner, "download_measurements", lambda _study: measurements)
    monkeypatch.setattr(runner, "_run_evaluation", lambda *_args: calls.append(_args))

    runner.run_simulation(study, "temporal")

    assert len(calls) == 1
    assert (Path(study["output_root"]) / "sim/a100/instance_a/temporal/DONE").is_file()


def test_failed_simulation_does_not_write_done(tmp_path, monkeypatch):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    study["latency_seeds"] = [0]
    measurements = tmp_path / "download"
    _write_measurement(measurements, "a100", "instance_a")
    monkeypatch.setattr(runner, "download_measurements", lambda _study: measurements)
    monkeypatch.setattr(
        runner,
        "_run_evaluation",
        lambda *_args: _fail("simulation failed"),
    )

    with pytest.raises(RuntimeError, match="simulation failed"):
        runner.run_simulation(study, "temporal")

    assert not (Path(study["output_root"]) / "sim/a100/instance_a/temporal/DONE").exists()


def test_simulation_resume_skips_completed_latency_seeds(tmp_path, monkeypatch):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    study["latency_seeds"] = [0, 1]
    measurements = tmp_path / "download"
    _write_measurement(measurements, "a100", "instance_a")
    method_dir = Path(study["output_root"]) / "sim/a100/instance_a/temporal"
    (method_dir / "0").mkdir(parents=True)
    (method_dir / "0/DONE").touch()
    (method_dir / "1").mkdir()
    (method_dir / "1/stale").touch()
    calls = []
    monkeypatch.setattr(runner, "download_measurements", lambda _study: measurements)
    monkeypatch.setattr(runner, "_run_evaluation", lambda *_args: calls.append(_args))

    runner.run_simulation(study, "temporal", resume=True)

    assert len(calls) == 1
    assert calls[0][2].name == "1"
    assert not (method_dir / "1/stale").exists()
    assert (method_dir / "DONE").is_file()

    runner.run_simulation(study, "temporal", resume=True)
    assert len(calls) == 1


def test_publish_uploads_done_after_study_data(tmp_path, monkeypatch):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    unit = tmp_path / "unit"
    unit.mkdir()
    (unit / "profile.json").write_text("{}\n", encoding="utf-8")
    (unit / "DONE").touch()
    calls = []

    class FakeApi:
        def upload_folder(self, **kwargs):
            calls.append(("folder", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("file", kwargs))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))

    sync.publish_study_unit(study, unit, Path("measure/a100/instance_a"))

    assert [kind for kind, _ in calls] == ["folder", "file"]
    assert calls[0][1]["ignore_patterns"] == ["DONE"]
    assert calls[1][1]["path_in_repo"].endswith("/measure/a100/instance_a/DONE")


def test_failed_data_upload_does_not_upload_done(tmp_path, monkeypatch):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    unit = tmp_path / "unit"
    unit.mkdir()
    (unit / "DONE").touch()
    calls = []

    class FakeApi:
        def upload_folder(self, **kwargs):
            calls.append(("folder", kwargs))
            raise RuntimeError("upload failed")

        def upload_file(self, **kwargs):
            calls.append(("file", kwargs))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))

    with pytest.raises(RuntimeError, match="upload failed"):
        sync.publish_study_unit(study, unit, Path("measure/a100/instance_a"))

    assert [kind for kind, _ in calls] == ["folder"]


def test_protocol_dispatches_the_direct_study_commands(monkeypatch):
    study = {"study_id": "study"}
    calls = []
    monkeypatch.setattr(protocol, "load_study", lambda _path: study)
    monkeypatch.setattr(
        protocol,
        "run_measure",
        lambda value, hardware, *, resume: calls.append(("measure", value, hardware, resume)),
    )
    monkeypatch.setattr(
        protocol,
        "run_simulation",
        lambda value, method, *, resume: calls.append(("sim", value, method, resume)),
    )
    monkeypatch.setattr(protocol, "publish_measure", lambda value, hardware: calls.append(("publish-measure", value, hardware)))
    monkeypatch.setattr(protocol, "publish_simulation", lambda value, method: calls.append(("publish-sim", value, method)))

    protocol.main(["measure", "--spec", "study.yaml", "--hardware", "a100", "--resume"])
    protocol.main(["sim", "--spec", "study.yaml", "--method", "iid", "--resume"])
    protocol.main(["publish", "--spec", "study.yaml", "--stage", "measure", "--hardware", "l40s"])
    protocol.main(["publish", "--spec", "study.yaml", "--stage", "sim", "--method", "temporal"])

    assert calls == [
        ("measure", study, "a100", True),
        ("sim", study, "iid", True),
        ("publish-measure", study, "l40s"),
        ("publish-sim", study, "temporal"),
    ]


def test_study_specs_have_one_policy_and_no_simulation_host(tmp_path):
    real1 = _study(tmp_path, "cross_device_1real_3sim.yaml")
    real10 = _study(tmp_path, "cross_device_10real_20sim.yaml")

    assert real1["sessions"] == {"fit": [0], "evaluation": [0]}
    assert real10["sessions"] == {
        "fit": [0, 1, 2, 3, 4],
        "evaluation": [5, 6, 7, 8, 9],
    }
    assert real10["policy"]["id"] == "openvla-flappy"
    assert "rows" not in real10
    assert "experiments" not in real10
    assert "sim_host_device" not in real10


@pytest.mark.parametrize(
    "script",
    [
        "measure.sh",
        "sim.sh",
        "publish.sh",
        "start.sh",
        "measure_10.sh",
        "sim_20.sh",
        "sim_iid_20.sh",
        "upload_10.sh",
    ],
)
def test_study_shell_entrypoints_are_valid_bash(script):
    subprocess.run(["bash", "-n", script], check=True)


def test_numbered_entrypoints_reuse_the_generic_study_scripts():
    assert "SPEC=configs/experiments/cross_device_10real_20sim.yaml bash measure.sh" in Path(
        "measure_10.sh"
    ).read_text(encoding="utf-8")
    assert "METHOD=temporal bash sim.sh" in Path("sim_20.sh").read_text(encoding="utf-8")
    assert "METHOD=iid bash sim.sh" in Path("sim_iid_20.sh").read_text(encoding="utf-8")
    assert "bash publish.sh measure" in Path("upload_10.sh").read_text(encoding="utf-8")
    start = Path("start.sh").read_text(encoding="utf-8")
    assert "bash measure.sh" in start
    assert "bash publish.sh measure" in start
    assert "bash sim.sh" in start
    assert "bash publish.sh sim" in start
