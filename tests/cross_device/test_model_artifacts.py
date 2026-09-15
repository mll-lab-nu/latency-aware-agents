"""Artifact boundary contracts and snapshot containment."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from latency_bench.cross_device.profile_resolver import (
    resolve_model_artifacts,
    resolve_profile_artifacts,
)
import latency_bench.cross_device.protocol as protocol
from latency_bench.cross_device.spec import load_profile_spec


def _spec(tmp_path: Path) -> dict:
    return {
        "model_artifact": {
            "local_root": str(tmp_path / "default"),
            "checkpoint": "checkpoints/final.pt",
            "hf_repo": "org/models",
            "hf_path": "zero-latency/task/vla/model-v1",
        }
    }


def test_task_presets_define_model_and_runtime_artifact_coordinates():
    inverted_pendulum = load_profile_spec(
        "configs/profiles.yaml", "qwenoft", "inverted_pendulum"
    )
    mikasa = load_profile_spec(
        "configs/profiles.yaml", "qwenoft", "mikasa_intercept_grab_fast"
    )
    balance = load_profile_spec(
        "configs/profiles.yaml", "qwengr00t", "g1_sonic_balance_simple"
    )
    parkour = load_profile_spec(
        "configs/profiles.yaml", "qwengr00t", "extreme_parkour_go2"
    )
    haic = load_profile_spec("configs/profiles.yaml", "qwenoft", "haic_pull_cart")

    assert inverted_pendulum["model_artifact"]["checkpoint"] == (
        "checkpoints/steps_2000_pytorch_model.pt"
    )
    assert mikasa["model_artifact"]["checkpoint"] == "checkpoints/model.pt"
    assert "starvla-qwengr00t" in balance["model_artifact"]["hf_path"]
    assert parkour["runtime_artifact"]["checkpoint"] == "model_14999.pt"
    assert haic["runtime_artifact"]["checkpoint"] == "checkpoint_final.pt"


@pytest.mark.parametrize(
    ("model_id", "workload_id", "hf_path"),
    [
        ("qwenoft", "mikasa_intercept_grab_fast", "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenoft-h1/mikasa_intercept_grab_fast_qwenoft_formal_latest_r8gpu8_20260907T082154Z"),
        ("qwenpi_v3", "mikasa_intercept_grab_fast", "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwenpi_v3-h1/mikasa_intercept_grab_fast_qwenpi_v3_formal_latest_r8gpu8_20260907T082154Z"),
        ("qwengr00t", "mikasa_intercept_grab_fast", "zero-latency/mikasa-intercept-grab-fast/vla/starvla-qwengr00t-h1/mikasa_intercept_grab_fast_qwengr00t_formal_latest_r8gpu7_20260907T075003Z"),
        ("wanoft", "mikasa_intercept_grab_fast", "zero-latency/mikasa-intercept-grab-fast/vla/starvla-wanoft-h1/mikasa_intercept_grab_fast_wanoft_formal_latest_r8gpu7_20260907T075003Z"),
        ("qwenoft", "g1_sonic_balance_simple", "zero-latency/humanoidbench-balance-simple/vla/starvla-qwenoft-h1/humanoidbench_balance_simple_qwenoft_formal_latest"),
        ("qwenpi_v3", "g1_sonic_balance_simple", "zero-latency/humanoidbench-balance-simple/vla/starvla-qwenpi_v3-h1/humanoidbench_balance_simple_qwenpi_v3_formal_latest"),
        ("qwengr00t", "g1_sonic_balance_simple", "zero-latency/humanoidbench-balance-simple/vla/starvla-qwengr00t-h1/humanoidbench_balance_simple_qwengr00t_formal_latest"),
        ("wanoft", "g1_sonic_balance_simple", "zero-latency/humanoidbench-balance-simple/vla/starvla-wanoft-h1/humanoidbench_balance_simple_wanoft_formal_latest"),
    ],
)
def test_starvla_h1_profile_artifacts(model_id, workload_id, hf_path):
    spec = load_profile_spec("configs/profiles.yaml", model_id, workload_id)

    assert spec["model_artifact"]["hf_path"] == hf_path
    assert spec["model_artifact"]["checkpoint"] == "checkpoints/model.pt"
    assert spec["model_artifact"]["model_config"] == "config.yaml"
    assert spec["session_count"] == 5
    assert spec["eval_episodes"] == 50
    assert spec["simulation_gpu"] == (
        "cuda:0" if workload_id == "mikasa_intercept_grab_fast" else None
    )


def test_local_model_root_resolves_exact_checkpoint_and_provenance(tmp_path):
    spec = _spec(tmp_path)
    checkpoint = tmp_path / "default/checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    resolved = resolve_model_artifacts(spec)

    assert resolved["checkpoint_path"] == str(checkpoint.resolve())
    assert resolved["model_provenance"] == {
        "source": "local",
        "repo_id": "org/models",
        "path_in_repo": "zero-latency/task/vla/model-v1",
        "checkpoint": "checkpoints/final.pt",
    }


def test_existing_local_root_with_missing_checkpoint_fails(tmp_path):
    spec = _spec(tmp_path)
    Path(spec["model_artifact"]["local_root"]).mkdir()

    with pytest.raises(FileNotFoundError):
        resolve_model_artifacts(spec)


def test_local_model_root_provenance_has_no_remote_revision(tmp_path):
    spec = _spec(tmp_path)
    checkpoint = tmp_path / "default/checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    resolved = resolve_model_artifacts(spec)

    assert resolved["model_provenance"]["source"] == "local"


@pytest.mark.parametrize("blob_location", ["repo", "ancestor", "other_repo"])
def test_missing_local_root_downloads_only_exact_hf_artifact(tmp_path, monkeypatch, blob_location):
    spec = _spec(tmp_path)
    cache = tmp_path / "models--org--models"
    snapshot = cache / "snapshots" / ("b" * 40)
    blob = {
        "repo": cache / "blobs/checkpoint",
        "ancestor": tmp_path / "blobs/checkpoint",
        "other_repo": tmp_path / "models--other--model/blobs/checkpoint",
    }[blob_location]
    blob.parent.mkdir(parents=True)
    blob.touch()
    checkpoint = snapshot / "zero-latency/task/vla/model-v1/checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.symlink_to(blob)
    calls = {}

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: (
                calls.update(snapshot=kwargs) or str(snapshot)
            ),
        ),
    )

    if blob_location != "repo":
        with pytest.raises(ValueError, match="escapes"):
            resolve_model_artifacts(spec)
        return
    resolved = resolve_model_artifacts(spec)

    assert resolved["checkpoint_path"] == str(checkpoint)
    assert calls["snapshot"]["repo_id"] == "org/models"
    assert calls["snapshot"]["repo_type"] == "model"
    assert calls["snapshot"]["allow_patterns"] == [
        "zero-latency/task/vla/model-v1/**"
    ]
    assert resolved["model_provenance"]["source"] == "hf"


def test_root_hf_artifact_downloads_from_snapshot_root(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec["model_artifact"]["hf_path"] = "."
    snapshot = tmp_path / "snapshot"
    (snapshot / "checkpoints").mkdir(parents=True)
    (snapshot / "checkpoints/final.pt").touch()
    calls = {}
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: (
                calls.update(snapshot=kwargs) or str(snapshot)
            )
        ),
    )

    resolved = resolve_model_artifacts(spec)

    assert resolved["checkpoint_path"] == str(snapshot / "checkpoints/final.pt")
    assert "allow_patterns" not in calls["snapshot"]


def test_profile_run_rejects_conflicting_artifact_sources():
    with pytest.raises(SystemExit):
        protocol.main([
            "run", "--model", "qwengr00t", "--workload", "extreme_parkour_go2",
            "--model-path", "/models/local", "--hf-artifacts",
        ])


def test_model_path_override_keeps_exact_relative_checkpoint(tmp_path):
    spec = _spec(tmp_path)
    override = tmp_path / "override"
    checkpoint = override / "checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    resolved = resolve_model_artifacts(spec, model_path=override)

    assert resolved["checkpoint_path"] == str(checkpoint.resolve())
    assert resolved["model_provenance"]["source"] == "local"


def test_explicit_hf_artifact_override_forces_download_even_with_local_root(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    local_root = Path(spec["model_artifact"]["local_root"])
    (local_root / "checkpoints").mkdir(parents=True)
    (local_root / "checkpoints/final.pt").touch()

    snapshot = tmp_path / "snapshot"
    checkpoint = snapshot / "alt/repo/weights/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    calls = {}
    spec["model_artifact"].update(
        {
            "hf_repo": "org/alt",
            "hf_path": "alt/repo",
            "checkpoint": "weights/final.pt",
        }
    )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: (
                calls.update(snapshot=kwargs) or str(snapshot)
            ),
        ),
    )

    resolved = resolve_model_artifacts(spec, hf_artifacts=True)

    assert resolved["checkpoint_path"] == str(checkpoint)
    assert calls["snapshot"]["repo_id"] == "org/alt"
    assert calls["snapshot"]["repo_type"] == "model"
    assert calls["snapshot"]["allow_patterns"] == ["alt/repo/**"]
    assert resolved["model_provenance"]["source"] == "hf"


def test_model_config_resolves_from_the_same_artifact_root(tmp_path):
    spec = _spec(tmp_path)
    spec["model_artifact"]["model_config"] = "config.full.yaml"
    root = Path(spec["model_artifact"]["local_root"])
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints/final.pt").touch()
    (root / "config.full.yaml").touch()

    resolved = resolve_model_artifacts(spec)

    assert resolved["model_config_path"] == str((root / "config.full.yaml").resolve())
    assert resolved["model_provenance"]["model_config"] == "config.full.yaml"


@pytest.mark.parametrize("registry_runtime,profile_runtime", [(False, True), (True, False)])
def test_simulation_resolves_the_model_artifact_recorded_by_the_profile(
    tmp_path, monkeypatch, registry_runtime, profile_runtime
):
    spec = _spec(tmp_path)
    snapshot = tmp_path / "snapshot"
    root = snapshot / "zero-latency/task/vla/model-v1"
    checkpoint = root / "checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (root / "profile-config.yaml").touch()
    calls = {}
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: (
                calls.update(snapshot=kwargs) or str(snapshot)
            )
        ),
    )
    profile = {
        "provenance": {
            "model_artifact": {
                "repo_id": "org/models",
                "path_in_repo": "zero-latency/task/vla/model-v1",
                "checkpoint": "checkpoints/final.pt",
                "model_config": "profile-config.yaml",
                "source": "hf",
            }
        }
    }
    if registry_runtime:
        spec["runtime_artifact"] = {"checkpoint": "not-the-profile-runtime.pt"}
    if profile_runtime:
        profile["provenance"]["runtime_artifact"] = {
            **profile["provenance"]["model_artifact"],
            "checkpoint": "runtime.pt",
        }
        (root / "runtime.pt").touch()

    resolved = resolve_profile_artifacts(spec, profile)

    assert resolved["checkpoint_path"] == str(checkpoint)
    # The replay must use the model config recorded by the profile, not the
    # (possibly changed) live registry value.
    assert resolved["model_config_path"] == str(root / "profile-config.yaml")
    assert "revision" not in calls["snapshot"]
    if profile_runtime:
        assert resolved["runtime_checkpoint_path"] == str(root / "runtime.pt")
    else:
        assert "runtime_checkpoint_path" not in resolved


def test_profile_artifact_paths_must_stay_under_the_snapshot_root(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: str(snapshot)
        ),
    )

    escaping = {
        "repo_id": "org/models",
        "path_in_repo": "../../elsewhere",
        "checkpoint": "weights.pt",
        "source": "hf",
    }
    with pytest.raises(ValueError):
        resolve_profile_artifacts(
            spec, {"provenance": {"model_artifact": escaping}}
        )

    absolute = {
        "repo_id": "org/models",
        "path_in_repo": "zero-latency/task",
        "checkpoint": "/etc/passwd",
        "source": "hf",
    }
    with pytest.raises(ValueError):
        resolve_profile_artifacts(
            spec, {"provenance": {"model_artifact": absolute}}
        )


def test_protocol_forwards_model_path(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        protocol,
        "load_profile_spec",
        lambda *_args: {"model_id": "task", "workload_id": "workload"},
    )
    monkeypatch.setattr(
        protocol,
        "resolve_model_artifacts",
        lambda spec, *, model_path, hf_artifacts: (
            captured.update(model_path=model_path, hf_artifacts=hf_artifacts) or dict(spec)
        ),
    )
    monkeypatch.setattr(
        protocol,
        "run_profile",
        lambda spec, **kwargs: (
            captured.update(spec=spec, kwargs=kwargs) or {"RUN_ID": "run_1"}
        ),
    )

    protocol.main(
        [
            "run",
            "--model",
            "task",
            "--workload",
            "workload",
            "--model-path",
            "/tmp/model",
            "--publish",
        ]
    )

    assert captured["model_path"] == Path("/tmp/model")
    assert captured["hf_artifacts"] is False
    assert captured["kwargs"] == {"resume_run_id": None, "publish": True}


def test_protocol_forwards_explicit_hf_artifact_overrides(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        protocol,
        "load_profile_spec",
        lambda *_args: {
            "model_id": "task",
            "workload_id": "workload",
            "model_artifact": {
                "local_root": "/tmp/local",
                "checkpoint": "checkpoints/default.pt",
                "hf_repo": "org/default",
                "hf_path": "default/path",
                "model_config": "config.default.yaml",
            },
        },
    )
    monkeypatch.setattr(
        protocol,
        "resolve_model_artifacts",
        lambda spec, *, model_path, hf_artifacts: (
            captured.update(
                model_path=model_path,
                hf_artifacts=hf_artifacts,
                artifact=dict(spec["model_artifact"]),
            )
            or dict(spec)
        ),
    )
    monkeypatch.setattr(
        protocol,
        "run_profile",
        lambda spec, **kwargs: (
            captured.update(spec=spec, kwargs=kwargs) or {"RUN_ID": "run_1"}
        ),
    )

    protocol.main(
        [
            "run",
            "--model",
            "task",
            "--workload",
            "workload",
            "--hf-repo-id",
            "org/override",
            "--hf-path",
            "runs/alt",
            "--hf-checkpoint",
            "checkpoints/final.pt",
            "--hf-model-config",
            "config.full.yaml",
        ]
    )

    assert captured["model_path"] is None
    assert captured["hf_artifacts"] is True
    assert captured["artifact"] == {
        "local_root": "/tmp/local",
        "checkpoint": "checkpoints/final.pt",
        "hf_repo": "org/override",
        "hf_path": "runs/alt",
        "model_config": "config.full.yaml",
    }
    assert captured["kwargs"] == {"resume_run_id": None, "publish": False}
