from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups


class _LanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(3, 2)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False) for _ in range(4)])
        self.norm = torch.nn.LayerNorm(2)


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = torch.nn.Linear(2, 2, bias=False)
        self.language_model = _LanguageModel()


class _HFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.lm_head = torch.nn.Linear(2, 3, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight


class _QwenInterface(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _HFModel()


class _VLA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen_vl_interface = _QwenInterface()
        self.action_model = torch.nn.Linear(2, 2, bias=False)


class _NonQwenVLA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.world_model = torch.nn.Linear(2, 2, bias=False)
        self.action_model = torch.nn.Linear(2, 2, bias=False)


def _cfg(freeze_vit: bool = True, freeze_tied_embedding: bool = False):
    return OmegaConf.create(
        {
            "trainer": {
                "freeze_modules": "",
                "freeze_vit": freeze_vit,
                "freeze_tied_embedding": freeze_tied_embedding,
                "freeze_llm_layers": [0, 1],
                "learning_rate": {
                    "base": 2.0e-5,
                    "qwen_vl_interface": 1.0e-5,
                    "action_model": 1.0e-4,
                },
            }
        }
    )


def _non_qwen_no_freeze_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "trainer": {
                "freeze_modules": "",
                "freeze_vit": False,
                "freeze_tied_embedding": False,
                "freeze_llm_layers": [],
                "learning_rate": {
                    "base": 2.0e-5,
                    "action_model": 1.0e-4,
                },
            }
        }
    )


def test_freeze_vit_and_llm_layers_are_independent() -> None:
    model = _VLA()

    TrainerUtils.freeze_vit_and_llm_layers(model, _cfg(freeze_vit=True))

    assert not model.qwen_vl_interface.model.model.visual.weight.requires_grad
    assert model.qwen_vl_interface.model.model.language_model.embed_tokens.weight.requires_grad
    assert model.qwen_vl_interface.model.lm_head.weight.requires_grad
    assert not model.qwen_vl_interface.model.model.language_model.layers[0].weight.requires_grad
    assert not model.qwen_vl_interface.model.model.language_model.layers[1].weight.requires_grad
    assert model.qwen_vl_interface.model.model.language_model.layers[2].weight.requires_grad
    assert model.qwen_vl_interface.model.model.language_model.layers[3].weight.requires_grad
    assert model.qwen_vl_interface.model.model.language_model.norm.weight.requires_grad
    assert model.action_model.weight.requires_grad


def test_freeze_tied_embedding_freezes_embed_tokens_and_lm_head() -> None:
    model = _VLA()

    TrainerUtils.freeze_vit_and_llm_layers(model, _cfg(freeze_tied_embedding=True))

    assert model.qwen_vl_interface.model.model.language_model.embed_tokens.weight is model.qwen_vl_interface.model.lm_head.weight
    assert not model.qwen_vl_interface.model.model.language_model.embed_tokens.weight.requires_grad
    assert not model.qwen_vl_interface.model.lm_head.weight.requires_grad
    assert model.qwen_vl_interface.model.model.language_model.layers[2].weight.requires_grad
    assert model.qwen_vl_interface.model.model.language_model.norm.weight.requires_grad
    assert model.action_model.weight.requires_grad


def test_freeze_llm_layers_without_freeze_vit_keeps_visual_trainable() -> None:
    model = _VLA()

    TrainerUtils.freeze_vit_and_llm_layers(model, _cfg(freeze_vit=False))

    assert model.qwen_vl_interface.model.model.visual.weight.requires_grad
    assert not model.qwen_vl_interface.model.model.language_model.layers[0].weight.requires_grad
    assert not model.qwen_vl_interface.model.model.language_model.layers[1].weight.requires_grad
    assert model.qwen_vl_interface.model.model.language_model.layers[2].weight.requires_grad
    assert model.action_model.weight.requires_grad


def test_optimizer_groups_exclude_frozen_vit_and_llm_params() -> None:
    model = _VLA()
    cfg = _cfg(freeze_vit=True, freeze_tied_embedding=True)

    TrainerUtils.freeze_vit_and_llm_layers(model, cfg)
    groups = build_param_lr_groups(model, cfg)
    optimizer_param_ids = {id(param) for group in groups for param in group["params"]}

    assert id(model.qwen_vl_interface.model.model.visual.weight) not in optimizer_param_ids
    assert id(model.qwen_vl_interface.model.model.language_model.embed_tokens.weight) not in optimizer_param_ids
    assert id(model.qwen_vl_interface.model.lm_head.weight) not in optimizer_param_ids
    assert id(model.qwen_vl_interface.model.model.language_model.layers[0].weight) not in optimizer_param_ids
    assert id(model.qwen_vl_interface.model.model.language_model.layers[2].weight) in optimizer_param_ids
    assert id(model.qwen_vl_interface.model.model.language_model.norm.weight) in optimizer_param_ids
    assert id(model.action_model.weight) in optimizer_param_ids


def test_optimizer_groups_do_not_require_qwen_interface_when_no_qwen_freeze_requested() -> None:
    model = _NonQwenVLA()
    cfg = _non_qwen_no_freeze_cfg()

    groups = build_param_lr_groups(model, cfg)
    optimizer_param_ids = {id(param) for group in groups for param in group["params"]}

    assert id(model.world_model.weight) in optimizer_param_ids
    assert id(model.action_model.weight) in optimizer_param_ids
