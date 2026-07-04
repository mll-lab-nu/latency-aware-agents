from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from latency_bench.cross_device import runner


def test_finalize_profile_artifacts_writes_hardware_fingerprint(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profiles" / "openvla-flappy" / "flappy" / "L_L40S"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.json").write_text(json.dumps({"profile_id": "L_L40S"}) + "\n", encoding="utf-8")
    fingerprint = {"gpu": {"name": "NVIDIA L40S"}, "packages": {"flash_attn": "2.7.4"}}
    monkeypatch.setattr(runner, "host_fingerprint", lambda: fingerprint)
    monkeypatch.delenv("HF_PROFILE_REPO", raising=False)
    manifest = {}
    task = SimpleNamespace(profile_output_dir=profile_dir)

    runner._finalize_profile_artifacts(task, tmp_path, manifest)

    assert json.loads((profile_dir / "hardware.json").read_text(encoding="utf-8")) == fingerprint
    assert not (profile_dir / "diagnostics.json").exists()
    profile = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
    assert profile["hardware"] == fingerprint
    assert profile["diagnostics_path"] == "hardware.json"


def test_subprocess_runtime_telemetry_is_nonblocking(tmp_path, monkeypatch):
    calls = []

    class FakeMonitor:
        def __init__(self, output_dir):
            assert output_dir == tmp_path

        def start(self):
            calls.append("start")
            return {"runtime_telemetry_enabled": True, "runtime_telemetry_path": "runtime_telemetry.jsonl"}

        def stop(self):
            calls.append("stop")
            return {"runtime_telemetry_summary_path": "runtime_telemetry_summary.json"}

    def fake_run(cmd, check):
        calls.append(("run", cmd, check))

    monkeypatch.setattr(runner, "RuntimeTelemetryMonitor", FakeMonitor)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    manifest = {}

    runner._run_subprocess_with_runtime_telemetry(["python", "-V"], tmp_path, manifest)

    assert calls == ["start", ("run", ["python", "-V"], True), "stop"]
    assert manifest["runtime_telemetry_enabled"] is True
    assert manifest["runtime_telemetry_path"] == "runtime_telemetry.jsonl"
    assert manifest["runtime_telemetry_summary_path"] == "runtime_telemetry_summary.json"


def test_subprocess_runtime_telemetry_stop_runs_when_command_fails(tmp_path, monkeypatch):
    calls = []

    class FakeMonitor:
        def __init__(self, output_dir):
            pass

        def start(self):
            calls.append("start")
            return {"runtime_telemetry_enabled": True}

        def stop(self):
            calls.append("stop")
            return {"runtime_telemetry_summary_path": "runtime_telemetry_summary.json"}

    def fake_run(cmd, check):
        calls.append("run")
        raise subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr(runner, "RuntimeTelemetryMonitor", FakeMonitor)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    try:
        runner._run_subprocess_with_runtime_telemetry(["python", "-V"], tmp_path, {})
    except subprocess.CalledProcessError:
        pass

    assert calls == ["start", "run", "stop"]
