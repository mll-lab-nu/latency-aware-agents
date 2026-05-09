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
    learner_module = types.ModuleType("sample_factory.algo.learning.learner")

    class Log:
        def warning(self, *args):
            pass

        def exception(self, *args):
            pass

    class Learner:
        @staticmethod
        def load_checkpoint(checkpoints, device):
            return {"original": (checkpoints, device)}

    learner_module.Learner = Learner
    learner_module.log = Log()

    monkeypatch.setitem(sys.modules, "sample_factory", sample_factory)
    monkeypatch.setitem(sys.modules, "sample_factory.algo", algo)
    monkeypatch.setitem(sys.modules, "sample_factory.algo.learning", learning)
    monkeypatch.setitem(sys.modules, "sample_factory.algo.learning.learner", learner_module)
    return Learner


def test_patch_sample_factory_checkpoint_loader_uses_trusted_torch_load(monkeypatch):
    from training.sample_factory_common import checkpoint_io

    Learner = _install_fake_sample_factory_learner(monkeypatch)
    monkeypatch.setattr(checkpoint_io, "_SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED", False, raising=False)
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
    from training.sample_factory_common import checkpoint_io

    Learner = _install_fake_sample_factory_learner(monkeypatch)
    monkeypatch.setattr(checkpoint_io, "_SAMPLE_FACTORY_CHECKPOINT_LOADER_PATCHED", False, raising=False)

    checkpoint_io.patch_sample_factory_checkpoint_loader()
    first_load_checkpoint = Learner.load_checkpoint
    checkpoint_io.patch_sample_factory_checkpoint_loader()

    assert Learner.load_checkpoint is first_load_checkpoint
