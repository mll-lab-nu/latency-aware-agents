from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from latency_bench.cross_device import protocol, runner
from latency_bench.cross_device.spec import load_protocol_spec
from latency_bench.utils.io import read_json, read_yaml, write_json


SPEC_PATH = Path("configs/experiments/cross_device_10real_20sim.yaml")


def _device(device_id: str) -> dict:
    return {
        "id": device_id,
        "display_name": device_id.upper(),
        "profile_id": f"L_{device_id.upper()}",
    }


def _spec(tmp_path: Path) -> dict:
    return {
        "protocol_id": "cross_device_10real_20sim_v1",
        "output_root": str(tmp_path / "protocol"),
        "checkpoint_kind": "best",
        "n_episodes": 100,
        "seed_set": [0],
        "latency_seed_set": list(range(20)),
        "latency_methods": ["temporal", "iid"],
        "measure_sessions": {
            "fit": [0, 1, 2, 3, 4],
            "heldout": [5, 6, 7, 8, 9],
        },
        "sim_host_device": "rtx5090",
        "target_devices": ["a100"],
        "hf_repo": "latency-sensitive-bench/profiles",
        "hf_repo_type": "dataset",
        "devices": [_device("a100"), _device("rtx5090")],
        "rows": [
            {
                "policy": "openvla-flappy",
                "env": "Flappy",
                "policy_type": "starvla",
                "model_type": "openvla",
                "env_name": "flappy",
            }
        ],
    }


def test_protocol_spec_inherits_base_and_defines_session_split():
    spec = load_protocol_spec(SPEC_PATH)

    assert spec["measure_sessions"] == {
        "fit": [0, 1, 2, 3, 4],
        "heldout": [5, 6, 7, 8, 9],
    }
    assert spec["latency_seed_set"] == list(range(20))
    assert spec["latency_methods"] == ["temporal", "iid"]
    assert spec["sim_parallel_envs"] == 1
    assert [row["policy"] for row in spec["rows"]] == ["openvla-flappy"]


def test_legacy_protocol_spec_loading_is_unchanged():
    path = Path("configs/experiments/cross_device_protocol.yaml")

    assert load_protocol_spec(path) == read_yaml(path)


def test_profile_build_id_uses_artifacts_fitter_not_manifest_timestamps(
    tmp_path: Path,
    monkeypatch,
):
    spec = _spec(tmp_path)
    fitter_sha256 = runner._profile_fitter_sha256()
    monkeypatch.setattr(runner, "git_sha", lambda: "revision_a")
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {"selected_env_fps": 16.0, "selected_obs_fps": 16.0})
    task = runner._protocol_task(
        spec=spec,
        row=spec["rows"][0],
        target_device=_device("a100"),
        target_instance_id="instance_a",
        target_instance_origin="gpu_uuid_hash",
        stage="measure",
        output_root=Path(spec["output_root"]),
        measure_session_id=0,
        measure_role="fit",
    )
    for filename in (
        "e2e_latencies.jsonl",
        "infer_latencies.jsonl",
        "episode_metrics.jsonl",
        "actions.jsonl",
        "steps.jsonl",
    ):
        task.output_dir.mkdir(parents=True, exist_ok=True)
        (task.output_dir / filename).write_text(f"{filename}\n", encoding="utf-8")

    first = runner._profile_build_id(spec, _device("a100"), "instance_a", calibration, [task])
    write_json(task.output_dir / "manifest.json", {"completed_at": "later"})
    second = runner._profile_build_id(spec, _device("a100"), "instance_a", calibration, [task])
    monkeypatch.setattr(runner, "_profile_fitter_sha256", lambda: "updated-fitter")
    fitter_changed = runner._profile_build_id(
        spec, _device("a100"), "instance_a", calibration, [task]
    )
    monkeypatch.setattr(runner, "_profile_fitter_sha256", lambda: fitter_sha256)
    (task.output_dir / "actions.jsonl").write_text("changed\n", encoding="utf-8")
    changed = runner._profile_build_id(spec, _device("a100"), "instance_a", calibration, [task])

    assert first == second
    assert fitter_changed != first
    assert changed != first


def test_measure_runs_fit_profile_then_heldout(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path)
    output_root = Path(spec["output_root"])
    calibration_path = output_root / "calibration/env_fps_calibration.json"
    write_json(calibration_path, {"selected_env_fps": 16.0, "selected_obs_fps": 16.0})
    events = []

    monkeypatch.setattr(
        runner,
        "detect_hardware",
        lambda spec, hardware: {"raw": "A100", "device": _device("a100")},
    )
    monkeypatch.setattr(runner, "host_fingerprint", lambda: {})
    monkeypatch.setattr(runner, "resolve_instance_id", lambda fingerprint: "instance_a")
    monkeypatch.setattr(runner, "resolve_instance_origin", lambda: "gpu_uuid_hash")
    monkeypatch.setattr(
        runner,
        "_calibrate_protocol_once",
        lambda **kwargs: calibration_path,
    )

    def run_sessions(**kwargs):
        profile_build_id = (
            kwargs["profile_build_id"] if "profile_build_id" in kwargs else None
        )
        events.append((kwargs["role"], tuple(kwargs["session_ids"]), profile_build_id))
        return [
            runner._protocol_task(
                spec=spec,
                row=spec["rows"][0],
                target_device=_device("a100"),
                target_instance_id="instance_a",
                target_instance_origin="gpu_uuid_hash",
                stage="measure",
                output_root=output_root,
                measure_session_id=session_id,
                measure_role=kwargs["role"],
                profile_build_id=profile_build_id,
            )
            for session_id in kwargs["session_ids"]
        ]

    def fit_profile(**kwargs):
        events.append(("profile", tuple(task.measure_session_id for task in kwargs["fit_tasks"]), None))
        profile_dir = output_root / "profile_builds/openvla/flappy/a100/instance_a/build_a"
        write_json(profile_dir / "profile.json", {"profile_fitter_sha256": "fitter_a"})
        return "build_a", profile_dir, "artifact_a"

    monkeypatch.setattr(runner, "_run_measure_sessions", run_sessions)
    monkeypatch.setattr(runner, "_fit_protocol_profile", fit_profile)

    runner.run_multi_session_measure(
        Namespace(out=None, hardware="auto", dry_run=False),
        spec,
    )

    assert events == [
        ("fit", (0, 1, 2, 3, 4), None),
        ("profile", (0, 1, 2, 3, 4), None),
        ("heldout", (5, 6, 7, 8, 9), "build_a"),
    ]
    completion = read_json(
        output_root / "completion/measure/a100/instance_a/build_a.json"
    )
    assert completion["fit_run_dirs"]
    assert completion["heldout_run_dirs"]
    assert completion["measure_sessions"] == spec["measure_sessions"]
    assert completion["n_episodes"] == 100
    assert completion["profile_artifact_sha256"] == "artifact_a"
    assert completion["profile_fitter_sha256"] == "fitter_a"

    runner.run_multi_session_measure(
        Namespace(out=None, hardware="auto", dry_run=False),
        spec,
    )
    assert len(events) == 3


def test_sim_runs_twenty_seeds_and_uploads_completion_last(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path)
    output_root = Path(spec["output_root"])
    download_root = tmp_path / "download"
    profile_path = output_root / "profile_builds/openvla/flappy/a100/instance_a/build_a/profile.json"
    (profile_path.parent / "latency_burst_model.json").parent.mkdir(parents=True, exist_ok=True)
    (profile_path.parent / "latency_burst_model.json").write_text("model\n", encoding="utf-8")
    artifact_sha256 = runner._profile_artifact_sha256(profile_path.parent)
    write_json(
        profile_path,
        {
            "profile_artifact_sha256": artifact_sha256,
            "profile_fitter_sha256": "fitter_a",
        },
    )
    measure_completion = (
        download_root
        / "raw/a100/instances/instance_a/cross_device/cross_device_10real_20sim_v1/"
        "completion/measure/a100/instance_a/build_a.json"
    )
    write_json(
        measure_completion,
        {
            "model_type": "openvla",
            "env_name": "flappy",
            "target_device_id": "a100",
            "target_profile_instance_id": "instance_a",
            "profile_build_id": "build_a",
            "profile_artifact_sha256": artifact_sha256,
            "profile_fitter_sha256": "fitter_a",
        },
    )
    tasks = []
    uploads = []
    monkeypatch.setattr(runner, "detect_hardware", lambda spec, hardware: {"raw": "5090", "device": _device("rtx5090")})
    monkeypatch.setattr(runner, "host_fingerprint", lambda: {})
    monkeypatch.setattr(runner, "resolve_instance_id", lambda fingerprint: "instance_5090")
    monkeypatch.setattr(runner, "download_measure_completions", lambda spec, destination: download_root)
    monkeypatch.setattr(runner, "download_profile_build", lambda spec, completion, output_root: profile_path)
    monkeypatch.setattr(runner, "_run_task", lambda task, *args, **kwargs: tasks.append(task))
    monkeypatch.setattr(runner, "publish_protocol_paths", lambda *args, **kwargs: uploads.append("artifacts"))
    monkeypatch.setattr(runner, "publish_protocol_completion", lambda *args, **kwargs: uploads.append("completion"))
    args = Namespace(out=None, hardware="auto", latency_method="temporal", resume=True)

    runner.run_all_profile_simulations(args, spec)

    assert [task.latency_seed for task in tasks] == list(range(20))
    assert {task.latency_method for task in tasks} == {"temporal"}
    assert {task.target_profile_path for task in tasks} == {profile_path}
    assert uploads[-1] == "completion"
    completion = read_json(
        output_root
        / "completion/sim/temporal/rtx5090/instance_5090/a100/instance_a/build_a.json"
    )
    assert completion["latency_seed_set"] == list(range(20))
    assert len(completion["run_dirs"]) == 20
    assert completion["seed_set"] == [0]
    assert completion["n_episodes"] == 100
    assert completion["profile_artifact_sha256"] == artifact_sha256
    assert completion["profile_fitter_sha256"] == "fitter_a"

    tasks.clear()
    (profile_path.parent / "latency_burst_model.json").write_text(
        "corrupted\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="profile artifact hash mismatch"):
        runner.run_all_profile_simulations(args, spec)
    assert tasks == []


def test_latency_methods_have_distinct_paths_and_share_profile(tmp_path: Path):
    spec = _spec(tmp_path)
    output_root = Path(spec["output_root"])
    profile_path = output_root / "profile_builds/openvla/flappy/a100/instance_a/build_a/profile.json"
    common = {
        "spec": spec,
        "row": spec["rows"][0],
        "target_device": _device("a100"),
        "target_instance_id": "instance_a",
        "target_instance_origin": "measure_completion",
        "stage": "sim_cross",
        "output_root": output_root,
        "latency_seed": 3,
        "sim_host_device": _device("rtx5090"),
        "sim_host_instance_id": "instance_5090",
        "target_profile_path": profile_path,
        "profile_build_id": "build_a",
    }

    temporal = runner._protocol_task(**common, latency_method="temporal")
    iid = runner._protocol_task(**common, latency_method="iid")

    assert temporal.target_profile_path == iid.target_profile_path
    assert temporal.output_dir != iid.output_dir
    assert temporal.generated_config_path != iid.generated_config_path
    assert temporal.output_dir.parent.name == "temporal"
    assert iid.output_dir.parent.name == "iid"


def test_sim_requires_every_target_device_completion(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path)
    spec["target_devices"] = ["a100", "l40s"]
    download_root = tmp_path / "download"
    write_json(
        download_root
        / "raw/a100/instances/instance_a/cross_device/cross_device_10real_20sim_v1/"
        "completion/measure/a100/instance_a/build_a.json",
        {
            "target_device_id": "a100",
            "target_profile_instance_id": "instance_a",
            "profile_build_id": "build_a",
            "profile_artifact_sha256": "artifact_a",
        },
    )
    monkeypatch.setattr(
        runner,
        "detect_hardware",
        lambda spec, hardware: {"raw": "5090", "device": _device("rtx5090")},
    )
    monkeypatch.setattr(runner, "host_fingerprint", lambda: {})
    monkeypatch.setattr(runner, "resolve_instance_id", lambda fingerprint: "instance_5090")
    monkeypatch.setattr(
        runner,
        "download_measure_completions",
        lambda spec, destination: download_root,
    )

    with pytest.raises(FileNotFoundError, match="l40s"):
        runner.run_all_profile_simulations(
            Namespace(out=None, hardware="auto", latency_method="temporal", resume=True),
            spec,
        )


def test_measure_upload_publishes_completion_after_artifacts(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path)
    output_root = Path(spec["output_root"])
    completion_path = output_root / "completion/measure/a100/instance_a/build_a.json"
    relative_dirs = [
        "calibration/openvla-flappy/a100/instance_a",
        "runs/measure/openvla-flappy/a100/instance_a/fit",
        "runs/measure/openvla-flappy/a100/instance_a/build_a/heldout",
        "generated_configs/measure/openvla-flappy/a100/instance_a",
    ]
    for relative_dir in relative_dirs:
        (output_root / relative_dir).mkdir(parents=True)
    profile_path = "profile_builds/openvla/flappy/a100/instance_a/build_a/profile.json"
    profile_dir = (output_root / profile_path).parent
    (profile_dir / "latency_burst_model.json").parent.mkdir(parents=True, exist_ok=True)
    (profile_dir / "latency_burst_model.json").write_text("model\n", encoding="utf-8")
    artifact_sha256 = runner._profile_artifact_sha256(profile_dir)
    write_json(
        output_root / profile_path,
        {
            "profile_artifact_sha256": artifact_sha256,
            "profile_fitter_sha256": "fitter_a",
        },
    )
    write_json(
        completion_path,
        {
            "model_type": "openvla",
            "env_name": "flappy",
            "target_device_id": "a100",
            "target_profile_instance_id": "instance_a",
            "profile_build_id": "build_a",
            "profile_artifact_sha256": artifact_sha256,
            "profile_fitter_sha256": "fitter_a",
            "profile_path": profile_path,
            "calibration_path": f"{relative_dirs[0]}/env_fps_calibration.json",
            "fit_run_dirs": [relative_dirs[1]],
            "heldout_run_dirs": [relative_dirs[2]],
            "generated_config_dirs": [relative_dirs[3]],
        },
    )
    uploads = []
    monkeypatch.setattr(runner, "detect_hardware", lambda spec, hardware: {"device": _device("a100")})
    monkeypatch.setattr(runner, "host_fingerprint", lambda: {})
    monkeypatch.setattr(runner, "resolve_instance_id", lambda fingerprint: "instance_a")
    monkeypatch.setattr(runner, "publish_protocol_paths", lambda *args, **kwargs: uploads.append("artifacts"))
    monkeypatch.setattr(runner, "publish_protocol_folder", lambda *args, **kwargs: uploads.append("profile"))
    monkeypatch.setattr(runner, "publish_protocol_completion", lambda *args, **kwargs: uploads.append("completion"))

    runner.publish_measure(Namespace(out=None, hardware="auto"), spec)

    assert uploads[-1] == "completion"


def test_measure_upload_failure_does_not_publish_completion(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path)
    output_root = Path(spec["output_root"])
    completion_path = output_root / "completion/measure/a100/instance_a/build_a.json"
    profile_path = output_root / "profile_builds/openvla/flappy/a100/instance_a/build_a/profile.json"
    (profile_path.parent / "latency_burst_model.json").parent.mkdir(parents=True, exist_ok=True)
    (profile_path.parent / "latency_burst_model.json").write_text("model\n", encoding="utf-8")
    artifact_sha256 = runner._profile_artifact_sha256(profile_path.parent)
    write_json(
        profile_path,
        {
            "profile_artifact_sha256": artifact_sha256,
            "profile_fitter_sha256": "fitter_a",
        },
    )
    write_json(
        completion_path,
        {
            "model_type": "openvla",
            "env_name": "flappy",
            "profile_build_id": "build_a",
            "profile_artifact_sha256": artifact_sha256,
            "profile_fitter_sha256": "fitter_a",
            "profile_path": "profile_builds/openvla/flappy/a100/instance_a/build_a/profile.json",
            "calibration_path": "calibration/a100/instance_a/env_fps_calibration.json",
            "fit_run_dirs": [],
            "heldout_run_dirs": [],
            "generated_config_dirs": [],
        },
    )
    completions = []
    monkeypatch.setattr(
        runner,
        "detect_hardware",
        lambda spec, hardware: {"device": _device("a100")},
    )
    monkeypatch.setattr(runner, "host_fingerprint", lambda: {})
    monkeypatch.setattr(runner, "resolve_instance_id", lambda fingerprint: "instance_a")
    monkeypatch.setattr(
        runner,
        "publish_protocol_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("upload failed")),
    )
    monkeypatch.setattr(
        runner,
        "publish_protocol_completion",
        lambda *args, **kwargs: completions.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        runner.publish_measure(Namespace(out=None, hardware="auto"), spec)

    assert completions == []


def test_convenience_scripts_use_protocol_entrypoint_and_measure_uploads():
    measure = Path("measure_10.sh").read_text(encoding="utf-8")
    upload = Path("upload_10.sh").read_text(encoding="utf-8")
    temporal = Path("sim_20.sh").read_text(encoding="utf-8")
    iid = Path("sim_iid_20.sh").read_text(encoding="utf-8")

    assert "latency_bench.cross_device.protocol measure" in measure
    assert "bash upload_10.sh" in measure
    assert "latency_bench.cross_device.protocol publish-measure" in upload
    assert "--latency-method temporal" in temporal
    assert "--latency-method iid" in iid


def test_aggregate_downloads_profiles_referenced_by_measure_completions(
    tmp_path: Path, monkeypatch
):
    spec = _spec(tmp_path)
    raw_root = tmp_path / "downloaded"
    completion_path = (
        raw_root
        / "raw/a100/instances/instance_a/cross_device"
        / spec["protocol_id"]
        / "completion/measure/a100/instance_a/build_a.json"
    )
    completion = {
        "protocol_id": spec["protocol_id"],
        "model_type": "openvla",
        "env_name": "flappy",
        "target_device_id": "a100",
        "target_profile_instance_id": "instance_a",
        "profile_build_id": "build_a",
    }
    write_json(completion_path, completion)
    downloads = []
    monkeypatch.setattr(protocol, "load_protocol_spec", lambda path: spec)
    monkeypatch.setattr(
        protocol,
        "hf_env",
        lambda: {
            "repo_id": spec["hf_repo"],
            "repo_type": spec["hf_repo_type"],
            "revision": None,
        },
    )
    monkeypatch.setattr(protocol, "download_results_tree", lambda **kwargs: raw_root)
    monkeypatch.setattr(
        protocol,
        "download_profile_build",
        lambda loaded_spec, loaded_completion, output_root: downloads.append(
            (loaded_spec, loaded_completion, output_root)
        ),
    )
    monkeypatch.setattr(protocol, "write_protocol_tables", lambda **kwargs: {})

    protocol.main(
        [
            "aggregate",
            "--spec",
            str(tmp_path / "spec.yaml"),
            "--root",
            str(tmp_path / "results"),
            "--out",
            str(tmp_path / "tables"),
            "--overwrite",
        ]
    )

    assert downloads == [(spec, completion, tmp_path / "results")]
