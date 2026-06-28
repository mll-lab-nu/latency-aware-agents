from __future__ import annotations

import sys
import types

import torch


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

    assert result == {"checkpoint": "latest.pth"}
    assert calls == [("latest.pth", torch.device("cpu"))]


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
