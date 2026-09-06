"""Long-lived artifact boundary contracts: pinned replay and snapshot containment."""

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
            "hf_revision": "a" * 40,
        }
    }


def test_task_presets_pin_model_and_runtime_artifact_coordinates():
    inverted_pendulum = load_profile_spec(
        "configs/profiles.yaml", "qwenoft", "inverted_pendulum_rgb_state"
    )
    mikasa = load_profile_spec(
        "configs/profiles.yaml", "qwenoft", "mikasa_intercept_grab_fast"
    )
    balance = load_profile_spec(
        "configs/profiles.yaml", "official_gr00t", "g1_sonic_balance_simple"
    )
    parkour = load_profile_spec(
        "configs/profiles.yaml", "official_gr00t", "extreme_parkour_go2"
    )
    haic = load_profile_spec("configs/profiles.yaml", "official_gr00t", "haic_pull_cart")

    assert inverted_pendulum["model_artifact"]["hf_revision"] == (
        "a374b83248b130dd1ad0294b2ee44906650dba0e"
    )
    assert inverted_pendulum["model_artifact"]["checkpoint"] == (
        "checkpoints/steps_2000_pytorch_model.pt"
    )
    assert mikasa["model_artifact"]["checkpoint"] == (
        "checkpoints/steps_5000_pytorch_model.pt"
    )
    assert balance["model_artifact"]["hf_path"].endswith("gr00t-n1.7-10k-v1")
    assert balance["model_artifact"]["hf_revision"] == (
        "204be41b83c892a1c5d0b37a1dd0d628dd2c6a46"
    )
    assert parkour["runtime_artifact"]["checkpoint"] == "model_14999.pt"
    assert haic["runtime_artifact"]["checkpoint"] == "checkpoint_final.pt"


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
        "requested_revision": "a" * 40,
    }


def test_existing_local_root_with_missing_checkpoint_fails(tmp_path):
    spec = _spec(tmp_path)
    Path(spec["model_artifact"]["local_root"]).mkdir()

    with pytest.raises(FileNotFoundError):
        resolve_model_artifacts(spec)


def test_local_model_root_does_not_claim_a_remote_commit(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec["model_artifact"]["hf_revision"] = "main"
    checkpoint = tmp_path / "default/checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: SimpleNamespace(
                repo_info=lambda **_kwargs: SimpleNamespace(sha="sha_remote")
            )
        ),
    )

    resolved = resolve_model_artifacts(spec)

    assert resolved["model_provenance"]["source"] == "local"
    assert "resolved_revision" not in resolved["model_provenance"]


@pytest.mark.parametrize("blob_location", ["repo", "ancestor", "other_repo"])
def test_missing_local_root_downloads_only_exact_hf_artifact(tmp_path, monkeypatch, blob_location):
    spec = _spec(tmp_path)
    spec["model_artifact"]["hf_revision"] = "main"
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
            HfApi=lambda: SimpleNamespace(
                repo_info=lambda **kwargs: (
                    calls.update(repo_info=kwargs) or SimpleNamespace(sha="b" * 40)
                )
            ),
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
    assert calls["repo_info"] == {
        "repo_id": "org/models",
        "repo_type": "model",
        "revision": "main",
    }
    assert calls["snapshot"]["repo_id"] == "org/models"
    assert calls["snapshot"]["repo_type"] == "model"
    assert calls["snapshot"]["revision"] == "b" * 40
    assert calls["snapshot"]["allow_patterns"] == [
        "zero-latency/task/vla/model-v1/**"
    ]
    assert resolved["model_provenance"]["source"] == "hf"
    assert resolved["model_provenance"]["requested_revision"] == "main"
    assert resolved["model_provenance"]["resolved_revision"] == "b" * 40


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


@pytest.mark.parametrize("source,revision", [("hf", "main"), ("hf", "abc123"), ("local", "a" * 40)])
def test_profile_replay_rejects_unpinned_artifacts(tmp_path, source, revision):
    with pytest.raises(ValueError, match="fixed 40-hex commit"):
        resolve_profile_artifacts(_spec(tmp_path), {"provenance": {"model_artifact": {
            "source": source, "resolved_revision": revision,
        }}})


def test_profile_run_rejects_conflicting_artifact_sources():
    with pytest.raises(SystemExit):
        protocol.main([
            "run", "--model", "official_gr00t", "--workload", "extreme_parkour_go2",
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
def test_simulation_resolves_the_model_revision_recorded_by_the_profile(
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
                "resolved_revision": "c" * 40,
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
    assert calls["snapshot"]["revision"] == "c" * 40
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
        "resolved_revision": "c" * 40,
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
        "resolved_revision": "c" * 40,
        "source": "hf",
    }
    with pytest.raises(ValueError):
        resolve_profile_artifacts(
            spec, {"provenance": {"model_artifact": absolute}}
        )


def test_protocol_forwards_model_path_and_session_python(monkeypatch):
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
            "--session-python",
            "/tmp/python",
            "--publish",
        ]
    )

    assert captured["model_path"] == Path("/tmp/model")
    assert captured["hf_artifacts"] is False
    assert captured["spec"]["session_python"] == "/tmp/python"
    assert captured["kwargs"] == {"resume_run_id": None, "publish": True}


def test_simulation_requires_session_python_for_native_entrypoint(monkeypatch):
    monkeypatch.setattr(
        protocol,
        "load_profile_spec",
        lambda *_args: {
            "model_id": "task",
            "workload_id": "workload",
            "eval_entrypoint": "native_eval.py",
        },
    )

    with pytest.raises(SystemExit):
        protocol.main(
            [
                "simulate",
                "--model",
                "task",
                "--workload",
                "workload",
                "--gpu",
                "1x-gpu",
                "--instance",
                "instance_abc",
                "--method",
                "zero",
                "--seeds",
                "0",
            ]
        )


def test_profile_run_requires_session_python_for_native_entrypoint(monkeypatch):
    monkeypatch.setattr(
        protocol,
        "load_profile_spec",
        lambda *_args: {
            "model_id": "task",
            "workload_id": "workload",
            "eval_entrypoint": "native_eval.py",
        },
    )
    monkeypatch.setattr(protocol, "resolve_model_artifacts", lambda spec, **kwargs: spec)

    with pytest.raises(SystemExit):
        protocol.main(
            [
                "run",
                "--model",
                "task",
                "--workload",
                "workload",
            ]
        )
