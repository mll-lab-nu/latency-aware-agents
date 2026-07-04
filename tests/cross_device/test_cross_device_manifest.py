from __future__ import annotations

import json
import sys

from latency_bench.cross_device.models import ProtocolTask, RuntimeSelection
from latency_bench.cross_device import manifest


def _task(tmp_path, *, stage: str = "sim_same") -> ProtocolTask:
    profile_path = tmp_path / "profiles" / "profile.json"
    return ProtocolTask(
        stage=stage,
        experiment_id="exp2_same_device_calibration",
        experiment_ids=["exp2_same_device_calibration"],
        panel="P2",
        row={
            "policy": "OpenVLA",
            "env": "flappy",
            "policy_type": "vla",
            "model_type": "openvla",
            "env_name": "flappy",
            "runtime_env": "openvla",
        },
        target_device={"id": "L_L40S", "display_name": "L40S", "profile_id": "l40s"},
        replay_device=None,
        seed=7,
        seed_set=[7],
        n_episodes=3,
        checkpoint_kind="best",
        output_dir=tmp_path / "runs" / stage,
        generated_config_path=tmp_path / "configs" / f"{stage}.yaml",
        target_profile_path=profile_path if stage != "measure" else None,
        profile_output_dir=profile_path.parent if stage == "measure" else None,
        instance_id="instance_abc",
        instance_origin="gpu_uuid_hash",
        target_profile_instance_id="instance_abc",
        runtime_env_alias="openvla",
        runtime_conda_env="openvla",
    )


def _runtime() -> RuntimeSelection:
    return RuntimeSelection(
        requested="auto",
        detected_raw="openvla",
        matched_alias="openvla",
        matched_conda_env="openvla",
        selected_alias="openvla",
        allow_all=False,
        allow_mismatch=False,
    )


def test_build_base_manifest_records_protocol_runtime_and_host_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "now_iso", lambda: "2026-07-04T00:00:00+00:00")
    monkeypatch.setattr(manifest, "git_sha", lambda: "abc123")
    monkeypatch.setattr(manifest, "host_fingerprint", lambda: {"gpu": {"name": "L40S"}})
    monkeypatch.setattr(
        manifest,
        "runtime_process_metadata",
        lambda: {
            "python_executable": sys.executable,
            "cuda_visible_devices": "0",
            "torch_version": "test",
            "torch_cuda_version": "12.1",
        },
    )
    task = _task(tmp_path)

    payload = manifest.build_base_manifest(
        task,
        {"protocol_id": "cross-device-v1"},
        {"device": {"id": "L_L40S"}, "raw": {"gpu": "L40S"}},
        _runtime(),
        ["python", "-m", "latency_bench.run", "--output-dir", str(task.output_dir)],
        None,
    )

    assert payload["protocol_id"] == "cross-device-v1"
    assert payload["stage"] == "sim_same"
    assert payload["experiment_ids"] == ["exp2_same_device_calibration"]
    assert payload["target_device_id"] == "L_L40S"
    assert payload["target_profile_instance_id"] == "instance_abc"
    assert payload["gpu_uuid_hash"] == "abc"
    assert payload["runtime_env_matched_alias"] == "openvla"
    assert payload["hardware_fingerprint"] == {"gpu": {"name": "L40S"}}
    assert payload["cuda_visible_devices"] == "0"
    assert payload["git_sha"] == "abc123"
    assert payload["status"] == "running"
    assert payload["command"].startswith("python -m latency_bench.run")


def test_record_skipped_task_writes_per_run_and_aggregate_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "now_iso", lambda: "2026-07-04T00:00:00+00:00")
    monkeypatch.setattr(manifest, "git_sha", lambda: "abc123")
    monkeypatch.setattr(manifest, "host_fingerprint", lambda: {"gpu": {"name": "L40S"}})
    monkeypatch.setattr(
        manifest,
        "runtime_process_metadata",
        lambda: {
            "python_executable": sys.executable,
            "cuda_visible_devices": None,
            "torch_version": "test",
            "torch_cuda_version": None,
        },
    )
    task = _task(tmp_path)
    out = tmp_path / "cross_device"

    manifest.record_skipped_task(
        task,
        {"protocol_id": "cross-device-v1"},
        {"device": {"id": "L_L40S"}, "raw": {"gpu": "L40S"}},
        _runtime(),
        "missing profile",
        out,
    )

    per_run = json.loads((task.output_dir / "manifest.json").read_text(encoding="utf-8"))
    aggregate = [
        json.loads(line)
        for line in (out / "manifests" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert per_run["status"] == "skipped"
    assert per_run["skip_reason"] == "missing profile"
    assert aggregate == [per_run]


def test_completed_manifest_exists_only_accepts_completed_status(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    assert manifest.completed_manifest_exists(output_dir) is False

    (output_dir / "manifest.json").write_text(json.dumps({"status": "failed"}) + "\n", encoding="utf-8")
    assert manifest.completed_manifest_exists(output_dir) is False

    (output_dir / "manifest.json").write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")
    assert manifest.completed_manifest_exists(output_dir) is True
