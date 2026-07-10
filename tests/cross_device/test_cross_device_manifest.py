from __future__ import annotations

import json
import sys
from pathlib import Path

from latency_bench.utils.io import write_jsonl
from latency_bench.cross_device.planning import ProtocolTask
from latency_bench.cross_device import manifest


SPEC = {
    "protocol_id": "cross-device-v1",
}


def _task(tmp_path, *, stage: str = "sim_same") -> ProtocolTask:
    profile_path = tmp_path / "profiles" / "profile.json"
    return ProtocolTask(
        stage=stage,
        experiment_ids=["exp2_same_device_calibration"],
        row={
            "policy": "OpenVLA",
            "env": "flappy",
            "policy_type": "vla",
            "model_type": "openvla",
            "env_name": "flappy",
        },
        target_device={"id": "L_L40S", "display_name": "L40S", "profile_id": "l40s"},
        replay_device=None,
        seed=7,
        seed_set=[7],
        latency_seed=None if stage == "measure" else 11,
        latency_seed_set=[11, 13],
        n_episodes=3,
        checkpoint_kind="best",
        output_dir=tmp_path / "runs" / stage,
        generated_config_path=tmp_path / "configs" / f"{stage}.yaml",
        target_profile_path=profile_path if stage != "measure" else None,
        profile_output_dir=profile_path.parent if stage == "measure" else None,
        instance_id="instance_abc",
        instance_origin="gpu_uuid_hash",
        target_profile_instance_id="instance_abc",
    )


def test_build_base_manifest_records_host_metadata(tmp_path, monkeypatch):
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
        SPEC,
        {"device": {"id": "L_L40S"}, "raw": {"gpu": "L40S"}},
        ["python", "-m", "latency_bench.run", "--output-dir", str(task.output_dir)],
        None,
    )

    assert payload["protocol_id"] == "cross-device-v1"
    assert payload["stage"] == "sim_same"
    assert payload["experiment_ids"] == ["exp2_same_device_calibration"]
    assert payload["target_device_id"] == "L_L40S"
    assert payload["target_profile_instance_id"] == "instance_abc"
    assert payload["gpu_uuid_hash"] == "abc"
    assert payload["hardware_fingerprint"] == {"gpu": {"name": "L40S"}}
    assert payload["cuda_visible_devices"] == "0"
    assert payload["git_sha"] == "abc123"
    assert payload["seed"] == 7
    assert payload["seed_set"] == [7]
    assert payload["latency_seed"] == 11
    assert payload["latency_seed_set"] == [11, 13]
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
        SPEC,
        {"device": {"id": "L_L40S"}, "raw": {"gpu": "L40S"}},
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


def _run_record(
    *,
    stage: str = "measure",
    experiment_ids: list[str] | None = None,
    status: str = "completed",
    output_dir: Path,
    profile_path: Path | None = None,
    completed_at: str = "2026-07-04T00:00:01+00:00",
    skip_reason: str = "",
    instance_id: str = "instance_a100",
    target_profile_instance_id: str = "instance_a100",
    replay_device_id: str | None = None,
) -> dict:
    return {
        "stage": stage,
        "experiment_ids": experiment_ids or ["exp2_same_device_calibration"],
        "policy": "OpenVLA",
        "env": "flappy",
        "target_device_id": "L_L40S",
        "target_profile_instance_id": target_profile_instance_id,
        "replay_device_id": replay_device_id,
        "instance_id": instance_id,
        "seed": 7,
        "status": status,
        "skip_reason": skip_reason,
        "notes": "",
        "output_dir": str(output_dir),
        "profile_path": "" if profile_path is None else str(profile_path),
        "generated_config_path": str(output_dir / "config.yaml"),
        "started_at": "2026-07-04T00:00:00+00:00",
        "completed_at": completed_at,
    }


def test_load_run_manifests_remaps_raw_paths_and_keeps_latest_completed_run(tmp_path):
    raw_root = tmp_path / "raw"
    protocol_root = raw_root / "L_L40S" / "instances" / "instance_a100" / "cross_device"
    old_root = Path("results/cross_device")
    output_dir = protocol_root / "runs" / "measure" / "OpenVLA" / "seed_7"
    profile_path = protocol_root / "profiles" / "openvla" / "flappy" / "L_L40S" / "instance_a100" / "profile.json"
    records = [
        _run_record(
            output_dir=old_root / output_dir.relative_to(protocol_root),
            profile_path=old_root / profile_path.relative_to(protocol_root),
            completed_at="2026-07-04T00:00:01+00:00",
        ),
        _run_record(
            output_dir=old_root / output_dir.relative_to(protocol_root),
            profile_path=old_root / profile_path.relative_to(protocol_root),
            completed_at="2026-07-04T00:00:02+00:00",
        ),
        _run_record(
            status="skipped",
            output_dir=old_root / output_dir.relative_to(protocol_root),
            profile_path=old_root / profile_path.relative_to(protocol_root),
            skip_reason="missing profile",
        ),
    ]
    write_jsonl(protocol_root / "manifests" / "runs.jsonl", records)

    loaded = manifest.load_run_manifests(tmp_path / "standard", raw_root=raw_root)

    completed = [record for record in loaded if record["status"] == "completed"]
    skipped = [record for record in loaded if record["status"] == "skipped"]
    assert len(completed) == 1
    assert completed[0]["completed_at"] == "2026-07-04T00:00:02+00:00"
    assert completed[0]["output_dir"] == str(output_dir)
    assert completed[0]["profile_path"] == str(profile_path)
    assert [record["skip_reason"] for record in skipped] == ["missing profile"]


def test_load_run_manifests_keeps_distinct_latency_seeds_for_same_eval_seed(tmp_path):
    out = tmp_path / "results"
    first = _run_record(stage="sim_cross", output_dir=out / "best_seed_7_latency_seed_11_n3")
    first["latency_seed"] = 11
    second = _run_record(stage="sim_cross", output_dir=out / "best_seed_7_latency_seed_13_n3")
    second["latency_seed"] = 13
    write_jsonl(out / "manifests" / "runs.jsonl", [first, second])

    loaded = manifest.load_run_manifests(out)

    assert {record["output_dir"] for record in loaded} == {
        first["output_dir"],
        second["output_dir"],
    }


def test_load_run_manifests_dedupes_legacy_and_explicit_same_latency_seed(tmp_path):
    out = tmp_path / "results"
    legacy = _run_record(
        stage="sim_cross",
        output_dir=out / "best_seed_7_n3",
        completed_at="2026-07-04T00:00:01+00:00",
    )
    current = _run_record(
        stage="sim_cross",
        output_dir=out / "best_seed_7_latency_seed_7_n3",
        completed_at="2026-07-04T00:00:02+00:00",
    )
    current["latency_seed"] = 7
    write_jsonl(out / "manifests" / "runs.jsonl", [legacy, current])

    loaded = manifest.load_run_manifests(out)

    assert len(loaded) == 1
    assert loaded[0]["output_dir"] == current["output_dir"]


def test_manifest_matching_and_instance_queries_use_completed_run_contract(tmp_path):
    completed = _run_record(
        stage="sim_cross",
        experiment_ids=["exp3_cross_device_prediction"],
        output_dir=tmp_path / "sim_cross",
        instance_id="instance_replay",
        replay_device_id="H100",
    )
    skipped = _run_record(
        stage="sim_cross",
        experiment_ids=["exp3_cross_device_prediction"],
        status="skipped",
        output_dir=tmp_path / "sim_cross_skipped",
        instance_id="instance_replay",
        replay_device_id="H100",
        skip_reason="missing profile",
    )
    manifests = [skipped, completed]
    row = {"policy": "OpenVLA", "env": "flappy"}

    assert manifest.matching_manifests(
        manifests,
        stage="sim_cross",
        experiment_id="exp3_cross_device_prediction",
        row=row,
        target_device_id="L_L40S",
        target_profile_instance_id="instance_a100",
        replay_device_id="H100",
        replay_host_instance_id="instance_replay",
    ) == [completed]
    assert manifest.matching_skipped_notes(
        manifests,
        stage="sim_cross",
        experiment_id="exp3_cross_device_prediction",
        row=row,
        target_device_id="L_L40S",
        target_profile_instance_id="instance_a100",
        replay_device_id="H100",
        replay_host_instance_id="instance_replay",
    ) == "missing profile"
    assert manifest.profile_instances_for_cross_pair(
        manifests,
        experiment_id="exp3_cross_device_prediction",
        row=row,
        target_device_id="L_L40S",
        replay_device_id="H100",
    ) == ["instance_a100"]
    assert manifest.replay_host_instances_for_cross_pair(
        manifests,
        experiment_id="exp3_cross_device_prediction",
        row=row,
        target_device_id="L_L40S",
        target_profile_instance_id="instance_a100",
        replay_device_id="H100",
    ) == ["instance_replay"]
