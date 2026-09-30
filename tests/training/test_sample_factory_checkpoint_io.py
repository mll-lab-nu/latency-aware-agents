from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from latency_bench.integrations.sample_factory import runtime as sf_runtime


def test_checkpoint_snapshot_survives_concurrent_pruning(monkeypatch, tmp_path):
    # Long-lived regression: checkpoint IO owns learner/evaluator pruning races.
    from training.common import checkpoints as checkpoint_io

    removed = tmp_path / "checkpoint_1.pth"
    retained = tmp_path / "checkpoint_2.pth"
    removed.touch()
    retained.touch()
    original_stat = Path.stat

    def stat_during_prune(path, *args, **kwargs):
        if path == removed:
            removed.unlink()
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_during_prune)
    assert checkpoint_io.checkpoint_mtimes(tmp_path) == {
        retained: original_stat(retained).st_mtime_ns,
    }


def _install_fake_sample_factory_learner(monkeypatch):
    sample_factory = types.ModuleType("sample_factory")
    sample_factory.__path__ = []
    algo = types.ModuleType("sample_factory.algo")
    algo.__path__ = []
    learning = types.ModuleType("sample_factory.algo.learning")
    learning.__path__ = []
    runners = types.ModuleType("sample_factory.algo.runners")
    runners.__path__ = []
    learner_module = types.ModuleType("sample_factory.algo.learning.learner")
    learner_worker_module = types.ModuleType("sample_factory.algo.learning.learner_worker")
    runner_parallel_module = types.ModuleType("sample_factory.algo.runners.runner_parallel")

    class Log:
        def warning(self, *args):
            pass

        def exception(self, *args):
            pass

    class Learner:
        @staticmethod
        def load_checkpoint(checkpoints, device):
            return {"original": (checkpoints, device)}

        def _save_impl(self, name_prefix, name_suffix, keep_checkpoints, verbose=True):
            return "original"

    def init_learner_process(sf_context, learner_worker):
        return ("original_init", sf_context, learner_worker)

    learner_module.Learner = Learner
    learner_module.log = Log()
    learner_worker_module.init_learner_process = init_learner_process
    runner_parallel_module.init_learner_process = init_learner_process

    monkeypatch.setitem(sys.modules, "sample_factory", sample_factory)
    monkeypatch.setitem(sys.modules, "sample_factory.algo", algo)
    monkeypatch.setitem(sys.modules, "sample_factory.algo.learning", learning)
    monkeypatch.setitem(sys.modules, "sample_factory.algo.runners", runners)
    monkeypatch.setitem(sys.modules, "sample_factory.algo.learning.learner", learner_module)
    monkeypatch.setitem(sys.modules, "sample_factory.algo.learning.learner_worker", learner_worker_module)
    monkeypatch.setitem(sys.modules, "sample_factory.algo.runners.runner_parallel", runner_parallel_module)
    return Learner, learner_worker_module, runner_parallel_module


def _reset_patch_flags(monkeypatch):
    from training.common import checkpoints as checkpoint_io

    monkeypatch.setattr(checkpoint_io, "_SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED", False, raising=False)
    monkeypatch.setattr(checkpoint_io, "_SAMPLE_FACTORY_CHECKPOINT_RETENTION_PATCHED", False, raising=False)
    monkeypatch.setattr(checkpoint_io, "_SAMPLE_FACTORY_LEARNER_INIT_PATCHED", False, raising=False)


def test_policy_runtime_loads_explicit_checkpoint(monkeypatch):
    from training.common import checkpoints as checkpoint_io

    checkpoint = {"model": {"weight": torch.tensor([1.0])}, "train_step": 12}
    calls = []
    monkeypatch.setattr(
        checkpoint_io,
        "trusted_torch_load",
        lambda path, device: calls.append((path, device)) or checkpoint,
    )
    loaded = []
    actor_critic = types.SimpleNamespace(load_state_dict=loaded.append)
    device = torch.device("cpu")

    result, resolved_path = sf_runtime.load_sample_factory_policy_state_dict(
        types.SimpleNamespace(),
        actor_critic,
        device,
        "teacher.pth",
    )

    assert result is checkpoint
    assert resolved_path == Path("teacher.pth").resolve()
    assert calls == [(Path("teacher.pth").resolve(), device)]
    assert loaded == [checkpoint["model"]]


@pytest.mark.parametrize(("kind", "pattern"), [("latest", "checkpoint_*"), ("best", "best_*")])
def test_policy_runtime_selects_checkpoint_kind(monkeypatch, kind, pattern):
    from training.common import checkpoints as checkpoint_io

    Learner, _learner_worker_module, _runner_parallel_module = _install_fake_sample_factory_learner(monkeypatch)
    calls = []
    monkeypatch.setattr(
        Learner,
        "checkpoint_dir",
        staticmethod(lambda cfg, policy_id: calls.append(("dir", policy_id)) or "/checkpoints"),
        raising=False,
    )
    monkeypatch.setattr(
        Learner,
        "get_checkpoints",
        staticmethod(lambda root, glob: calls.append((root, glob)) or ["candidate.pth"]),
        raising=False,
    )
    checkpoint = {"model": {"weight": torch.tensor([2.0])}}
    monkeypatch.setattr(
        checkpoint_io,
        "load_sample_factory_checkpoint_from_paths",
        lambda paths, device: calls.append((paths, device)) or (checkpoint, Path("/checkpoints/best_1.pth")),
    )
    loaded = []
    actor_critic = types.SimpleNamespace(load_state_dict=loaded.append)
    device = torch.device("cpu")

    result, resolved_path = sf_runtime.load_sample_factory_policy_state_dict(
        types.SimpleNamespace(load_checkpoint_kind=kind, policy_index=3),
        actor_critic,
        device,
        None,
    )

    assert result is checkpoint
    assert resolved_path == Path("/checkpoints/best_1.pth")
    assert calls == [
        ("dir", 3),
        ("/checkpoints", pattern),
        (["candidate.pth"], device),
    ]
    assert loaded == [checkpoint["model"]]


def test_patch_sample_factory_checkpoint_loader_uses_trusted_torch_load(monkeypatch):
    from training.common import checkpoints as checkpoint_io

    Learner, _learner_worker_module, _runner_parallel_module = _install_fake_sample_factory_learner(monkeypatch)
    _reset_patch_flags(monkeypatch)
    calls = []

    def trusted_load(path, device):
        calls.append((path, device))
        return {"checkpoint": path}

    monkeypatch.setattr(checkpoint_io, "trusted_torch_load", trusted_load)

    checkpoint_io.patch_sample_factory_checkpoint_loader()
    result = Learner.load_checkpoint(["old.pth", "latest.pth"], torch.device("cpu"))

    assert result == {"checkpoint": Path("latest.pth").resolve()}
    assert calls == [(Path("latest.pth").resolve(), torch.device("cpu"))]


def test_patch_sample_factory_checkpoint_loader_is_idempotent(monkeypatch):
    from training.common import checkpoints as checkpoint_io

    Learner, _learner_worker_module, runner_parallel_module = _install_fake_sample_factory_learner(monkeypatch)
    _reset_patch_flags(monkeypatch)

    checkpoint_io.patch_sample_factory_checkpoint_loader()
    first_load_checkpoint = Learner.load_checkpoint
    first_init = runner_parallel_module.init_learner_process
    checkpoint_io.patch_sample_factory_checkpoint_loader()

    assert Learner.load_checkpoint is first_load_checkpoint
    assert runner_parallel_module.init_learner_process is first_init


def test_patch_sample_factory_checkpoint_loader_patches_parallel_learner_init(monkeypatch):
    from training.common import checkpoints as checkpoint_io

    _Learner, _learner_worker_module, runner_parallel_module = _install_fake_sample_factory_learner(monkeypatch)
    _reset_patch_flags(monkeypatch)
    original_init = runner_parallel_module.init_learner_process

    checkpoint_io.patch_sample_factory_checkpoint_loader()

    assert runner_parallel_module.init_learner_process is checkpoint_io.init_learner_process_with_checkpoint_io_patch
    assert runner_parallel_module.init_learner_process is not original_init


def test_learner_child_init_wrapper_installs_checkpoint_io_before_original_init(monkeypatch):
    from training.common import checkpoints as checkpoint_io

    Learner, learner_worker_module, _runner_parallel_module = _install_fake_sample_factory_learner(monkeypatch)
    _reset_patch_flags(monkeypatch)
    original_save_impl = Learner._save_impl
    calls = []

    def init_learner_process(sf_context, learner_worker):
        calls.append(
            {
                "sf_context": sf_context,
                "learner_worker": learner_worker,
                "uses_original_save_impl": Learner._save_impl is original_save_impl,
            }
        )
        return "initialized"

    learner_worker_module.init_learner_process = init_learner_process

    result = checkpoint_io.init_learner_process_with_checkpoint_io_patch("ctx", "worker")

    assert result == "initialized"
    assert calls == [
        {
            "sf_context": "ctx",
            "learner_worker": "worker",
            "uses_original_save_impl": False,
        }
    ]


@pytest.mark.parametrize("missing", [False, True])
def test_selected_checkpoint_failure_does_not_load_older_state(tmp_path, monkeypatch, missing):
    from training.common import checkpoints as checkpoint_io

    old, latest = tmp_path / "old.pth", tmp_path / "latest.pth"
    torch.save({"model": {"weight": torch.ones(1)}, "train_step": 1}, old)
    if not missing:
        latest.write_bytes(b"invalid checkpoint")
    with pytest.raises(Exception) as expected:
        checkpoint_io.trusted_torch_load(latest, "cpu")
    with pytest.raises(type(expected.value)):
        checkpoint_io.load_sample_factory_checkpoint_from_paths([old, latest], "cpu")
    Learner, _, _ = _install_fake_sample_factory_learner(monkeypatch)
    _reset_patch_flags(monkeypatch)
    checkpoint_io.patch_sample_factory_checkpoint_loader()
    with pytest.raises(type(expected.value)):
        Learner.load_checkpoint([old, latest], "cpu")


def test_learner_initialization_distinguishes_fresh_training_and_resume(monkeypatch, tmp_path):
    from training.common import checkpoints as checkpoint_io

    Learner, _, _ = _install_fake_sample_factory_learner(monkeypatch)
    _reset_patch_flags(monkeypatch)
    checkpoint_io.patch_sample_factory_checkpoint_loader()
    learner = Learner()
    learner.is_initialized = False
    learner.cfg = types.SimpleNamespace(restart_behavior="overwrite", load_checkpoint_kind="latest")
    learner.device = "cpu"
    learner.checkpoint_dir = lambda cfg, policy_id: tmp_path
    learner.get_checkpoints = lambda root, pattern: []
    restored = []
    learner._load_state = lambda state, load_progress: restored.append((state, load_progress))
    learner.load_from_checkpoint(0)
    assert restored == []
    learner.cfg.restart_behavior = "resume"
    with pytest.raises(FileNotFoundError, match="No checkpoints"):
        learner.load_from_checkpoint(0)
    checkpoint = tmp_path / "checkpoint.pth"
    state = {"model": {"w": 1}, "optimizer": {"step": 7}, "train_step": 7, "env_steps": 99}
    torch.save(state, checkpoint)
    learner.get_checkpoints = lambda root, pattern: [checkpoint]
    learner.load_from_checkpoint(0)
    assert restored == [(state, True)]
