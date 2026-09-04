from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from latency_bench.cross_device.profile_resolver import resolve_model_artifacts
import latency_bench.cross_device.protocol as protocol
from latency_bench.cross_device.spec import load_profile_spec


def _spec(tmp_path: Path) -> dict:
    return {
        "model_artifact": {
            "local_root": str(tmp_path / "default"),
            "checkpoint": "checkpoints/final.pt",
            "hf_repo": "org/models",
            "hf_path": "zero-latency/task/vla/model-v1",
            "hf_revision": "main",
        }
    }


def test_task_presets_pin_model_and_runtime_artifact_coordinates():
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

    assert mikasa["model_artifact"]["checkpoint"] == (
        "checkpoints/steps_5000_pytorch_model.pt"
    )
    assert balance["model_artifact"]["hf_path"].endswith("gr00t-n1.7-15k-v1")
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
        "requested_revision": "main",
    }


def test_existing_local_root_with_missing_checkpoint_fails(tmp_path):
    spec = _spec(tmp_path)
    Path(spec["model_artifact"]["local_root"]).mkdir()

    with pytest.raises(FileNotFoundError):
        resolve_model_artifacts(spec)


def test_missing_local_root_downloads_only_exact_hf_artifact(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    snapshot = tmp_path / "snapshot"
    checkpoint = snapshot / "zero-latency/task/vla/model-v1/checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    calls = {}

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: SimpleNamespace(
                repo_info=lambda **kwargs: (
                    calls.update(repo_info=kwargs) or SimpleNamespace(sha="sha_123")
                )
            ),
            snapshot_download=lambda **kwargs: (
                calls.update(snapshot=kwargs) or str(snapshot)
            ),
        ),
    )

    resolved = resolve_model_artifacts(spec)

    assert resolved["checkpoint_path"] == str(checkpoint.resolve())
    assert calls["repo_info"] == {
        "repo_id": "org/models",
        "repo_type": "model",
        "revision": "main",
    }
    assert calls["snapshot"]["repo_id"] == "org/models"
    assert calls["snapshot"]["repo_type"] == "model"
    assert calls["snapshot"]["revision"] == "sha_123"
    assert calls["snapshot"]["allow_patterns"] == [
        "zero-latency/task/vla/model-v1/**"
    ]
    assert resolved["model_provenance"]["source"] == "hf"
    assert resolved["model_provenance"]["requested_revision"] == "main"
    assert resolved["model_provenance"]["resolved_revision"] == "sha_123"


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
        lambda spec, *, model_path: (
            captured.update(model_path=model_path) or dict(spec)
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
    assert captured["spec"]["session_python"] == "/tmp/python"
    assert captured["kwargs"] == {"resume_run_id": None, "publish": True}
