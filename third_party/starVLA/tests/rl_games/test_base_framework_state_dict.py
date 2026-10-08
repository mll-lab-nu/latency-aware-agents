from __future__ import annotations

import logging
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

rich_module = types.ModuleType("rich")
rich_logging_module = types.ModuleType("rich.logging")


class _FakeRichHandler(logging.StreamHandler):
    def __init__(self, *args, **kwargs):
        super().__init__()


rich_logging_module.RichHandler = _FakeRichHandler
rich_module.__spec__ = ModuleSpec("rich", loader=None)
rich_logging_module.__spec__ = ModuleSpec("rich.logging", loader=None)
rich_module.logging = rich_logging_module
sys.modules.setdefault("rich", rich_module)
sys.modules.setdefault("rich.logging", rich_logging_module)

from starVLA.model.framework.base_framework import _load_state_dict_allowing_qwen_tied_lm_head


class _TinyQwenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(3, 2)
        self.lm_head = torch.nn.Linear(2, 3, bias=False)
        self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


class _QwenInterface(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyQwenModel()


class _Framework(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen_vl_interface = _QwenInterface()
        self.action_model = torch.nn.Linear(2, 2)


def test_load_state_dict_accepts_missing_tied_qwen_lm_head_weight() -> None:
    source = _Framework()
    with torch.no_grad():
        source.qwen_vl_interface.model.embed.weight.fill_(3.0)
        source.action_model.weight.fill_(5.0)
    state_dict = dict(source.state_dict())
    del state_dict["qwen_vl_interface.model.lm_head.weight"]
    target = _Framework()

    _load_state_dict_allowing_qwen_tied_lm_head(target, state_dict)

    assert torch.equal(target.qwen_vl_interface.model.embed.weight, source.qwen_vl_interface.model.embed.weight)
    assert torch.equal(target.qwen_vl_interface.model.lm_head.weight, source.qwen_vl_interface.model.embed.weight)
    assert torch.equal(target.action_model.weight, source.action_model.weight)


def test_load_state_dict_rejects_unrelated_missing_keys() -> None:
    source = _Framework()
    state_dict = dict(source.state_dict())
    del state_dict["action_model.weight"]
    target = _Framework()

    with pytest.raises(RuntimeError, match="action_model.weight"):
        _load_state_dict_allowing_qwen_tied_lm_head(target, state_dict)


def test_load_state_dict_rejects_unexpected_keys() -> None:
    source = _Framework()
    state_dict = dict(source.state_dict())
    state_dict["unexpected.weight"] = torch.ones(1)
    target = _Framework()

    with pytest.raises(RuntimeError, match="unexpected.weight"):
        _load_state_dict_allowing_qwen_tied_lm_head(target, state_dict)
