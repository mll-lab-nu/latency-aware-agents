"""Long-lived model source and artifact filesystem boundary contracts."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from latency_bench.policy.model_source import model_file, resolve_model_source


def _config(root, source="local"):
    return {
        "policy": {
            "type": "custom",
            "source": source,
            "root": str(root),
            "checkpoint": "checkpoints/model.pt",
        },
        "env": {},
    }


def _model(root):
    checkpoint = root / "checkpoints/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    return checkpoint


def test_local_model_root_resolves_exact_checkpoint_and_provenance(tmp_path):
    checkpoint = _model(tmp_path)
    resolved = resolve_model_source(_config(tmp_path))
    assert model_file(resolved["policy"], "checkpoint") == str(checkpoint)
    assert resolved["provenance"]["model_artifact"] == {
        "source": "local",
        "local_root": str(tmp_path),
        "checkpoint": "checkpoints/model.pt",
    }


def test_existing_local_root_with_missing_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model_source(_config(tmp_path))


@pytest.mark.parametrize(
    "location,pinned",
    [("repo", False), ("repo", True), ("ancestor", True), ("other_repo", True)],
)
def test_explicit_hf_source_downloads_exact_artifact(
    tmp_path, monkeypatch, location, pinned
):
    config = _config("bundle", "hf")
    config["policy"]["hf_repo"] = "org/models"
    if pinned:
        config["policy"]["revision"] = "b" * 40
    cache = tmp_path / "models--org--models"
    snapshot = cache / "snapshots" / ("b" * 40)
    blob = {
        "repo": cache / "blobs/model",
        "ancestor": tmp_path / "blobs/model",
        "other_repo": tmp_path / "models--other--model/blobs/model",
    }[location]
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"model")
    checkpoint = snapshot / "bundle/checkpoints/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.symlink_to(blob)
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: calls.append(kwargs) or str(snapshot),
        ),
    )
    if location != "repo":
        with pytest.raises(ValueError, match="escapes"):
            resolve_model_source(config)
        return
    resolved = resolve_model_source(config)
    assert model_file(resolved["policy"], "checkpoint") == str(checkpoint)
    assert checkpoint.read_bytes() == b"model"
    assert calls[0]["allow_patterns"] == ["bundle/**"]
    assert ("revision" in calls[0]) == pinned
    origin = resolved["provenance"]["model_artifact"]
    assert origin["revision"] == "b" * 40
    # Resolved local inputs retain their original pinned identity.
    assert resolve_model_source(resolved)["provenance"] == resolved["provenance"]


def test_root_hf_artifact_downloads_from_snapshot_root(tmp_path, monkeypatch):
    checkpoint = _model(tmp_path)
    config = _config(".", "hf")
    config["policy"]["hf_repo"] = "org/models"
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **kwargs: calls.append(kwargs) or str(tmp_path),
        ),
    )
    resolved = resolve_model_source(config)
    assert model_file(resolved["policy"], "checkpoint") == str(checkpoint)
    assert "allow_patterns" not in calls[0]


def test_model_config_resolves_from_the_same_artifact_root(tmp_path):
    _model(tmp_path)
    (tmp_path / "config.full.yaml").write_text("model: example\n")
    (tmp_path / "manifest.json").write_text("{}")
    config = _config(tmp_path)
    config["policy"].update(
        model_config="config.full.yaml", task_manifest_path="manifest.json"
    )
    resolved = resolve_model_source(config)
    assert model_file(resolved["policy"], "model_config") == str(
        tmp_path / "config.full.yaml"
    )
    assert resolved["policy"]["task_manifest_path"] == str(tmp_path / "manifest.json")


def test_profile_artifact_paths_must_stay_under_the_snapshot_root(tmp_path):
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"model")
    root = tmp_path / "root"
    root.mkdir()
    config = _config(root)
    config["policy"]["checkpoint"] = "../outside.pt"
    with pytest.raises(ValueError, match="escapes"):
        resolve_model_source(config)


@pytest.mark.parametrize("source", ["local", "hf"])
def test_partial_file_override_preserves_other_model_inputs(
    tmp_path, monkeypatch, source
):
    """Model-source contract: an override retains config and metadata locations."""
    from latency_bench.policy.model_source import set_model_files

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "bundle"
    _model(root)
    (root / "config.full.yaml").write_text("{}")
    (root / "manifest.json").write_text("{}")
    config = _config("bundle", source)
    config["policy"].update(
        model_config="config.full.yaml", task_manifest_path="manifest.json"
    )
    if source == "hf":
        config["policy"]["hf_repo"] = "org/models"
        monkeypatch.setitem(
            sys.modules,
            "huggingface_hub",
            SimpleNamespace(snapshot_download=lambda **kwargs: str(tmp_path)),
        )
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"replacement")
    set_model_files(config, checkpoint=replacement)
    resolved = resolve_model_source(config)
    assert model_file(resolved["policy"], "checkpoint") == str(replacement)
    assert model_file(resolved["policy"], "model_config") == str(
        root / "config.full.yaml"
    )
    assert resolved["policy"]["task_manifest_path"] == str(root / "manifest.json")
