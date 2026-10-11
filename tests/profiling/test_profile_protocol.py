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
from latency_bench.profiling.cli import main
from latency_bench.profiling.runner import _profile_lock, _replace_current
from latency_bench.profiling.runtime_configs import measure_config
from latency_bench.profiling.spec import resolve_profile_config


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


def test_profile_coordinates_reject_path_traversal(profile_config):
    profile_config["profiling"]["model_id"] = "../escape"
    with pytest.raises(ValueError):
        resolve_profile_config(profile_config)


def test_profiling_rejects_ambiguous_sweep(profile_config):
    profile_config["evaluation"]["suites"] = [{"name": "fixed_0", "latency": {"method": "zero"}}]
    with pytest.raises(ValueError, match="sweeps"):
        resolve_profile_config(profile_config)


def test_cli_reads_the_same_run_config(profile_config, tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(profile_config))
    captured = {}
    monkeypatch.setattr("latency_bench.profiling.runner.run_profile", lambda config, **kwargs:
                        captured.update(config=config, kwargs=kwargs) or {"PROFILE_PATH": "/local/profile.json"})
    main(["run", "--config", str(path)])
    assert captured == {"config": profile_config, "kwargs": {"resume_run_id": None}}
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
    result = runner.run_profile(profile_config, resume_run_id="run_1")
    artifact_root = Path(profile_config["logging"]["output_dir"])
    ref = Path(result["PROFILE_REF"])
    assert Path(result["RUN_DIR"]) == artifact_root / "runs" / ref / "run_1"
    assert Path(result["PROFILE_PATH"]) == artifact_root / "profiles" / ref / "profile.json"
    assert calls == ["0", "1", "1"]
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
