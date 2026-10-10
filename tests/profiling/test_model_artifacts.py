"""Long-lived artifact source, relocation, and snapshot boundary contracts."""

from copy import deepcopy
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from latency_bench.profiling.profile_resolver import (
    resolve_model_artifacts,
)
from latency_bench.eval.config import resolve_evaluation_config


def _spec(tmp_path):
    return {
        "policy": {"type": "custom"}, "env": {},
        "profiling": {"model_artifact": {
            "source": "local", "local_root": str(tmp_path / "default"),
            "checkpoint": "checkpoints/final.pt", "hf_repo": "org/models",
            "hf_path": "zero-latency/task/vla/model-v1",
        }},
    }


def test_local_model_root_resolves_exact_checkpoint_and_provenance(tmp_path):
    spec = _spec(tmp_path)
    checkpoint = tmp_path / "default/checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    resolved = resolve_model_artifacts(spec)

    assert resolved["policy"]["checkpoint_path"] == str(checkpoint.resolve())
    assert resolved["provenance"]["model_artifact"] == {
        "source": "local",
        "local_root": str(tmp_path / "default"),
        "checkpoint": "checkpoints/final.pt",
    }


def test_existing_local_root_with_missing_checkpoint_fails(tmp_path):
    spec = _spec(tmp_path)
    Path(spec["profiling"]["model_artifact"]["local_root"]).mkdir()

    with pytest.raises(FileNotFoundError):
        resolve_model_artifacts(spec)


@pytest.mark.parametrize("blob_location,pinned", [
    ("repo", False), ("repo", True), ("ancestor", True), ("other_repo", True),
])
def test_explicit_hf_source_downloads_exact_artifact(tmp_path, monkeypatch, blob_location, pinned):
    spec = _spec(tmp_path)
    spec["profiling"]["model_artifact"]["source"] = "hf"
    if pinned:
        spec["profiling"]["model_artifact"]["revision"] = "b" * 40
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

    assert resolved["policy"]["checkpoint_path"] == str(checkpoint)
    assert calls["snapshot"]["repo_id"] == "org/models"
    assert calls["snapshot"]["repo_type"] == "model"
    assert calls["snapshot"]["allow_patterns"] == [
        "zero-latency/task/vla/model-v1/**"
    ]
    assert resolved["provenance"]["model_artifact"]["source"] == "hf"
    assert resolved["provenance"]["model_artifact"]["revision"] == "b" * 40
    if pinned:
        assert calls["snapshot"]["revision"] == "b" * 40
        assert resolved["provenance"]["model_artifact"]["revision"] == "b" * 40
    # Long-lived artifact boundary: evaluation must retain the snapshot filename
    # and bundle directory used by checkpoint/config loaders.
    config = {
        "evaluation": {}, "executor": {}, "logging": {}, "latency": {},
        "policy": {"checkpoint_path": resolved["policy"]["checkpoint_path"]},
    }
    resolve_evaluation_config(config)
    assert config["policy"]["checkpoint_path"] == str(checkpoint)
    assert Path(config["policy"]["checkpoint_path"]).read_bytes() == blob.read_bytes()
    replay = resolve_model_artifacts(spec, profile={"provenance": {"model_artifact": resolved["provenance"]["model_artifact"]}})
    assert replay["policy"]["checkpoint_path"] == str(checkpoint)
    assert calls["snapshot"]["revision"] == "b" * 40


def test_root_hf_artifact_downloads_from_snapshot_root(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec["profiling"]["model_artifact"]["source"] = "hf"
    spec["profiling"]["model_artifact"]["hf_path"] = "."
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

    assert resolved["policy"]["checkpoint_path"] == str(snapshot / "checkpoints/final.pt")
    assert "allow_patterns" not in calls["snapshot"]


def test_model_config_resolves_from_the_same_artifact_root(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec["policy"]["type"] = "starvla"
    spec["profiling"]["model_artifact"]["model_config"] = "config.full.yaml"
    root = Path(spec["profiling"]["model_artifact"]["local_root"])
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints/final.pt").touch()
    (root / "config.full.yaml").write_text(
        "backbone:\n  repo_id: org/unused-backbone\n"
    )
    (root / "config.yaml").write_text(
        "backbone:\n  repo_id: org/backbone\n  revision: fixed-commit\n"
        "  allow_patterns: ['*.safetensors']\n"
    )
    downloads = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **kwargs: (
            downloads.append(kwargs) or str(tmp_path / "cache/snapshots/fixed-commit")
        )
    ))

    resolved = resolve_model_artifacts(spec)

    assert resolved["policy"]["model_config_path"] == str((root / "config.full.yaml").resolve())
    assert resolved["provenance"]["model_artifact"]["model_config"] == "config.full.yaml"
    assert resolved["policy"]["backbone_path"] == str(tmp_path / "cache/snapshots/fixed-commit")
    assert downloads == [{
        "repo_id": "org/backbone", "revision": "fixed-commit",
        "allow_patterns": ["*.safetensors"],
    }]


@pytest.mark.parametrize("configured_runtime,profile_runtime", [(False, True), (True, False)])
def test_simulation_resolves_the_model_artifact_recorded_by_the_profile(
    tmp_path, monkeypatch, configured_runtime, profile_runtime
):
    spec = _spec(tmp_path)
    spec["profiling"]["model_artifact"]["source"] = "hf"
    snapshot = tmp_path / "snapshot"
    root = snapshot / "zero-latency/task/vla/model-v1"
    checkpoint = root / "checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (root / "profile-config.yaml").touch()
    (root / "config.yaml").write_text("{}\n")
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
                "revision": "model-revision",
                "source": "hf",
            }
        }
    }
    if configured_runtime:
        spec["profiling"]["runtime_artifact"] = {"source": "hf", "checkpoint": "not-the-profile-runtime.pt"}
    if profile_runtime:
        profile["provenance"]["runtime_artifact"] = {
            **profile["provenance"]["model_artifact"],
            "checkpoint": "runtime.pt",
        }
        (root / "runtime.pt").touch()

    resolved = resolve_model_artifacts(spec, profile=profile)

    assert resolved["policy"]["checkpoint_path"] == str(checkpoint)
    # The replay must use the model config recorded by the profile, not the
    # (possibly changed) current config value.
    assert resolved["policy"]["model_config_path"] == str(root / "profile-config.yaml")
    assert calls["snapshot"]["revision"] == "model-revision"
    if profile_runtime:
        assert resolved["env"]["runtime_checkpoint_path"] == str(root / "runtime.pt")
    else:
        assert "runtime_checkpoint_path" not in resolved["env"]


def test_profile_artifact_paths_must_stay_under_the_snapshot_root(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec["profiling"]["model_artifact"]["source"] = "hf"
    snapshot = tmp_path / "snapshot"
    (snapshot / "zero-latency/task").mkdir(parents=True)
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
        "revision": "fixed",
    }
    with pytest.raises(ValueError):
        resolve_model_artifacts(
            spec, profile={"provenance": {"model_artifact": escaping}}
        )

    absolute = {
        "repo_id": "org/models",
        "path_in_repo": "zero-latency/task",
        "checkpoint": "/etc/passwd",
        "source": "hf",
        "revision": "fixed",
    }
    with pytest.raises(ValueError):
        resolve_model_artifacts(
            spec, profile={"provenance": {"model_artifact": absolute}}
        )
    spec["profiling"]["model_artifact"]["source"] = "locla"
    with pytest.raises(ValueError, match="Unknown artifact source"):
        resolve_model_artifacts(spec, profile={"provenance": {"model_artifact": absolute}})


def test_local_replay_resolves_relocated_artifact(local_model_config, tmp_path, monkeypatch):
    config = resolve_model_artifacts(local_model_config)
    profile = {"provenance": {"model_artifact": config["provenance"]["model_artifact"]}}
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **kwargs: pytest.fail("local source must stay local")
    ))
    relocated = tmp_path / "relocated"
    shutil.copytree(config["provenance"]["model_artifact"]["local_root"], relocated)
    inputs = deepcopy(local_model_config)
    inputs["profiling"]["model_artifact"]["local_root"] = str(relocated)
    replay = resolve_model_artifacts(inputs, profile=profile)
    assert replay["policy"]["checkpoint_path"] == str(relocated / "checkpoints/model.pt")
    assert replay["policy"]["model_config_path"] == str(relocated / "config.yaml")
    assert replay["provenance"]["model_artifact"]["source"] == "local"


def test_starvla_uses_ordinary_checkpoint_and_backbone_paths(profile_config, tmp_path):
    """Long-lived policy input contract owned by profiling."""
    profile_config["policy"].update(
        type="starvla", checkpoint_path=str(tmp_path / "checkpoints/model.pt"),
        backbone_path=str(tmp_path / "backbone"),
    )
    for profile in (None, {"provenance": {}}):
        resolved = resolve_model_artifacts(profile_config, profile=profile)
        assert resolved["policy"] == profile_config["policy"]
