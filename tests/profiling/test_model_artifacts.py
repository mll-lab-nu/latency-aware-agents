"""Artifact boundary contracts and snapshot containment."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from latency_bench.profiling.profile_resolver import (
    profile_input_identity,
    resolve_model_artifacts,
    resolve_profile_artifacts,
)
import latency_bench.profiling.cli as protocol
from latency_bench.profiling import runner
from latency_bench.eval.config import resolve_evaluation_config
from latency_bench.core.config import load_config
from latency_bench.profiling.spec import load_profile_spec


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

    assert inverted_pendulum["model_artifact"]["checkpoint"] == (
        "checkpoints/steps_2000_pytorch_model.pt"
    )
    assert mikasa["model_artifact"]["checkpoint"] == "checkpoints/model.pt"
    assert "starvla-qwengr00t" in balance["model_artifact"]["hf_path"]


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
        "checkpoint": "checkpoints/final.pt",
        "files_sha256": {"checkpoints/final.pt": hashlib.sha256(b"").hexdigest()},
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
@pytest.mark.parametrize("pinned", [False, True])
def test_missing_local_root_downloads_only_exact_hf_artifact(tmp_path, monkeypatch, blob_location, pinned):
    spec = _spec(tmp_path)
    if pinned:
        spec["model_artifact"]["revision"] = "b" * 40
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
    assert resolved["model_provenance"]["revision"] == "b" * 40
    if pinned:
        assert calls["snapshot"]["revision"] == "b" * 40
        assert resolved["model_provenance"]["revision"] == "b" * 40
    # Long-lived artifact boundary: evaluation must retain the snapshot filename
    # and bundle directory used by checkpoint/config loaders.
    config = {
        "evaluation": {}, "executor": {}, "logging": {}, "latency": {},
        "policy": {"checkpoint_path": resolved["checkpoint_path"]},
    }
    resolve_evaluation_config(config)
    assert config["policy"]["checkpoint_path"] == str(checkpoint)
    assert Path(config["policy"]["checkpoint_path"]).read_bytes() == blob.read_bytes()
    replay = resolve_profile_artifacts(spec, {"provenance": {"model_artifact": resolved["model_provenance"]}})
    assert replay["checkpoint_path"] == str(checkpoint)
    assert calls["snapshot"]["revision"] == "b" * 40


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
            "run", "--model", "qwengr00t", "--workload", "g1_sonic_balance_simple",
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


def test_model_config_resolves_from_the_same_artifact_root(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    spec["model_artifact"]["model_config"] = "config.full.yaml"
    root = Path(spec["model_artifact"]["local_root"])
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

    assert resolved["model_config_path"] == str((root / "config.full.yaml").resolve())
    assert resolved["model_provenance"]["model_config"] == "config.full.yaml"
    assert resolved["backbone_path"] == str(tmp_path / "cache/snapshots/fixed-commit")
    assert downloads == [{
        "repo_id": "org/backbone", "revision": "fixed-commit",
        "allow_patterns": ["*.safetensors"],
    }]


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
                "files_sha256": {
                    "checkpoints/final.pt": hashlib.sha256(b"").hexdigest(),
                    "profile-config.yaml": hashlib.sha256(b"").hexdigest(),
                    "config.yaml": hashlib.sha256(b"{}\n").hexdigest(),
                },
            }
        }
    }
    if registry_runtime:
        spec["runtime_artifact"] = {"checkpoint": "not-the-profile-runtime.pt"}
    if profile_runtime:
        profile["provenance"]["runtime_artifact"] = {
            **profile["provenance"]["model_artifact"],
            "checkpoint": "runtime.pt",
            "files_sha256": {
                "runtime.pt": hashlib.sha256(b"").hexdigest(),
                "profile-config.yaml": hashlib.sha256(b"").hexdigest(),
                "config.yaml": hashlib.sha256(b"{}\n").hexdigest(),
            },
        }
        (root / "runtime.pt").touch()

    resolved = resolve_profile_artifacts(spec, profile)

    assert resolved["checkpoint_path"] == str(checkpoint)
    # The replay must use the model config recorded by the profile, not the
    # (possibly changed) live registry value.
    assert resolved["model_config_path"] == str(root / "profile-config.yaml")
    assert calls["snapshot"]["revision"] == "model-revision"
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
        "revision": "fixed",
        "files_sha256": {},
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
        "revision": "fixed",
        "files_sha256": {},
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


def _resolved_run_spec(tmp_path):
    spec = load_profile_spec("configs/profiles.yaml", "openvla", "flappy")
    spec.update(_spec(tmp_path))
    spec["artifact_root"] = str(tmp_path / "results")
    root = Path(spec["model_artifact"]["local_root"])
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints/final.pt").write_bytes(b"measured weights")
    spec["model_artifact"]["model_config"] = "config.full.yaml"
    for name in ("config.yaml", "config.full.yaml", "dataset_statistics.json", "task_contract.json"):
        (root / name).write_text("{}\n")
    backbone = tmp_path / "backbone"
    backbone.mkdir()
    (backbone / "processor_config.json").write_text("{}\n")
    spec["backbone_path"] = str(backbone)
    base = load_config(spec["base_config"])
    for key in ("task_contract_path", "task_manifest_path", "latency_prompt_map_path"):
        path = tmp_path / f"{key}.json"
        path.write_text("{}\n")
        base["policy"][key] = str(path)
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    spec["base_config"] = str(base_path)
    return resolve_model_artifacts(spec)


@pytest.mark.parametrize("changed", [
    "checkpoints/final.pt", "config.yaml", "config.full.yaml",
    "dataset_statistics.json", "task_contract.json", "backbone",
    "task_contract_path", "task_manifest_path", "latency_prompt_map_path",
])
def test_resume_rejects_changed_input_bytes_at_same_path(tmp_path, changed):
    spec = _resolved_run_spec(tmp_path)
    deployment = {"gpu_class": "1x-test", "instance_id": "instance_test", "gpus": [{}]}
    source = tmp_path / "run"
    source.mkdir()
    (source / "run_contract.json").write_text(json.dumps(runner._run_contract(spec, deployment)))
    runner._assert_run_contract(spec, deployment, source)
    if changed == "backbone":
        path = Path(spec["backbone_path"]) / "processor_config.json"
    elif changed.endswith("_path"):
        path = tmp_path / f"{changed}.json"
    else:
        path = Path(spec["model_artifact_root"]) / changed
    path.write_bytes(b"replacement bytes")
    with pytest.raises(ValueError, match="run contract does not match"):
        runner._assert_run_contract(spec, deployment, source)


def test_local_profile_replay_requires_identical_bytes_without_hf_substitution(tmp_path, monkeypatch):
    spec = _resolved_run_spec(tmp_path)
    profile = {"provenance": {"model_artifact": spec["model_provenance"]}}
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **kwargs: pytest.fail("local profile must not substitute HF contents")
    ))
    copy = tmp_path / "relocated"
    shutil.copytree(spec["model_artifact_root"], copy)
    replay = resolve_profile_artifacts(spec, profile, model_path=copy)
    assert profile_input_identity(replay) == profile_input_identity(spec)
    (copy / "checkpoints/final.pt").write_bytes(b"other weights")
    with pytest.raises(ValueError, match="identity does not match"):
        resolve_profile_artifacts(spec, profile, model_path=copy)
    del profile["provenance"]["model_artifact"]["files_sha256"]
    with pytest.raises(ValueError, match="no verified model identity"):
        resolve_profile_artifacts(spec, profile)


def test_starvla_implicit_backbone_requires_explicit_identity(tmp_path):
    spec = _resolved_run_spec(tmp_path)
    del spec["backbone_path"]
    (Path(spec["model_artifact_root"]) / "config.yaml").write_text(
        "framework:\n  qwenvl:\n    base_vlm: org/mutable-backbone\n"
    )
    with pytest.raises(ValueError, match="requires --backbone-path"):
        resolve_model_artifacts(spec)
    base = load_config(spec["base_config"])
    base["policy"]["backbone_path"] = str(tmp_path / "backbone")
    Path(spec["base_config"]).write_text(json.dumps(base))
    assert resolve_model_artifacts(spec)["backbone_path"] == str(tmp_path / "backbone")


def test_backbone_identity_includes_directory_symlink_contents(tmp_path):
    spec = _resolved_run_spec(tmp_path)
    tokenizer = tmp_path / "shared-tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    (Path(spec["backbone_path"]) / "tokenizer").symlink_to(tokenizer, target_is_directory=True)
    identity = profile_input_identity(spec)
    assert identity["backbone"]["tokenizer/tokenizer.json"] == hashlib.sha256(b"{}").hexdigest()
    (tokenizer / "tokenizer.json").write_text('{"changed": true}')
    assert profile_input_identity(spec) != identity


def test_measurement_input_mutation_cannot_mark_a_session_done(tmp_path, monkeypatch):
    spec = _resolved_run_spec(tmp_path)
    deployment = {"gpu_class": "1x-test", "instance_id": "instance_test", "gpus": [{}]}
    source_ref = Path("openvla/1x-test/flappy/instance_test/run_test")
    source = Path(spec["artifact_root"]) / "runs" / source_ref
    source.mkdir(parents=True)
    calibration = {"selected_env_fps": 30, "selected_obs_fps": 30}
    (source / "calibration.json").write_text(json.dumps(calibration))
    (source / "calibration.DONE").touch()
    (source / "run_contract.json").write_text(json.dumps(runner._run_contract(spec, deployment, calibration)))
    monkeypatch.setattr(runner, "_run_evaluation", lambda *args:
                        Path(spec["checkpoint_path"]).write_bytes(b"changed during measurement"))
    with pytest.raises(ValueError, match="run contract does not match"):
        runner._measure_session(spec, deployment, calibration, source_ref, 0, resume=False)
    assert not (source / "sessions/0/DONE").exists()


def test_calibration_cannot_replace_the_initial_model_identity(tmp_path, monkeypatch):
    spec = _resolved_run_spec(tmp_path)
    original = profile_input_identity(spec)
    deployment = {"gpu_class": "1x-test", "instance_id": "instance_test", "gpus": [{}]}
    monkeypatch.setattr(runner, "_collect_deployment", lambda _: (deployment, {}))
    monkeypatch.setattr(runner, "_new_run_id", lambda: "run_test")

    def calibrate(*args):
        Path(spec["checkpoint_path"]).write_bytes(b"changed during calibration")
        return {"selected_env_fps": 30, "selected_obs_fps": 30}

    monkeypatch.setattr(runner, "_calibrate", calibrate)
    with pytest.raises(ValueError, match="changed during calibration"):
        runner.run_profile(spec, resume_run_id=None, publish=False)
    contract = next(Path(spec["artifact_root"]).rglob("run_contract.json"))
    assert json.loads(contract.read_text())["input_files_sha256"] == original
    assert not (contract.parent / "DONE").exists()


@pytest.mark.parametrize("changed", [None, "checkpoint_path", "backbone_path"])
def test_replay_checks_input_identity_for_each_seed(tmp_path, monkeypatch, changed):
    spec = _resolved_run_spec(tmp_path)
    profile = {
        "source_run_id": "measured", "model_id": "openvla", "workload_id": "flappy",
        "gpu_class": "1x-test", "instance_id": "instance_test", "env_fps": 30,
        "per_slot_summary": [{}],
        "provenance": {"model_artifact": spec["model_provenance"],
                       "input_files_sha256": profile_input_identity(spec)},
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    ref = Path("openvla/1x-test/flappy/instance_test")
    monkeypatch.setattr(runner, "resolve_remote_profile", lambda *args, **kwargs: (ref, profile_path))
    calls = []

    def evaluate(spec, config, output):
        calls.append(output)
        if changed is None:
            return
        path = Path(spec[changed])
        if changed == "backbone_path":
            path /= "processor_config.json"
        path.write_bytes(b"changed during evaluation")

    monkeypatch.setattr(runner, "_run_evaluation", evaluate)
    if changed is None:
        runner.run_simulation(spec, gpu_class="1x-test", instance_id="instance_test",
                              method="temporal", latency_seeds=[0, 1], publish=False)
        assert len(calls) == 2
        assert all((output / "DONE").exists() for output in calls)
        return
    with pytest.raises(ValueError, match="inputs differ"):
        runner.run_simulation(spec, gpu_class="1x-test", instance_id="instance_test",
                              method="temporal", latency_seeds=[0, 1], publish=False)
    assert len(calls) == 1
    assert not (calls[0] / "DONE").exists()
    assert not list(Path(spec["artifact_root"]).rglob("DONE"))


def test_protocol_forwards_explicit_backbone_and_replay_paths(monkeypatch):
    captured = {}
    monkeypatch.setattr(protocol, "load_profile_spec", lambda *args: {})
    monkeypatch.setattr(protocol, "resolve_model_artifacts", lambda spec, **kwargs: dict(spec))
    monkeypatch.setattr(protocol, "run_profile", lambda spec, **kwargs: captured.update(spec) or {})
    protocol.main(["run", "--model", "m", "--workload", "w", "--backbone-path", "/b"])
    assert captured["backbone_path"] == "/b"
    monkeypatch.setattr(protocol, "run_simulation", lambda spec, **kwargs: captured.update(kwargs) or {})
    protocol.main(["simulate", "--model", "m", "--workload", "w", "--gpu", "g",
                   "--instance", "i", "--method", "zero", "--seeds", "0",
                   "--model-path", "/m", "--backbone-path", "/b"])
    assert captured["model_path"] == Path("/m")
    assert captured["backbone_path"] == Path("/b")


def test_gr00t_profile_entrypoint_resolves_model_identity(tmp_path, monkeypatch):
    from scripts.gym_adapt import gr00t

    spec = _resolved_run_spec(tmp_path)
    root = Path(spec["model_artifact_root"])
    Path(spec["checkpoint_path"]).rename(root / "checkpoints/model.pt")
    base = load_config(spec["base_config"])
    base["policy"]["backbone_path"] = spec["backbone_path"]
    Path(spec["base_config"]).write_text(json.dumps(base))
    captured = {}
    monkeypatch.setattr(runner, "run_profile", lambda resolved, **kwargs: captured.update(resolved) or {})
    monkeypatch.setattr(sys, "argv", ["gr00t", "profile", "--task", "ant",
                                    "--config", spec["base_config"], "--bundle", str(root),
                                    "--output", str(tmp_path / "profile")])
    gr00t.main()
    assert captured["model_provenance"]["files_sha256"]["checkpoints/model.pt"] == hashlib.sha256(b"measured weights").hexdigest()
    assert captured["backbone_path"] == spec["backbone_path"]
