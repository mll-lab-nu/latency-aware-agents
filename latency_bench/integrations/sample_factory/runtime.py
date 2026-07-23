from __future__ import annotations

from typing import Any

import torch


def resolve_torch_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "gpu":
        return "cuda"
    return device


def resolve_sample_factory_device(device: str) -> str:
    if device == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "gpu"
    return device


def load_sample_factory_policy_state_dict(
    cfg: Any,
    actor_critic: Any,
    device: torch.device,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    from training.common.checkpoints import (
        load_sample_factory_checkpoint_from_paths,
        trusted_torch_load,
    )

    if checkpoint_path is not None:
        checkpoint = trusted_torch_load(checkpoint_path, device)
    else:
        from sample_factory.algo.learning.learner import Learner

        name_prefix = {"latest": "checkpoint", "best": "best"}[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(
            Learner.checkpoint_dir(cfg, cfg.policy_index),
            f"{name_prefix}_*",
        )
        checkpoint = load_sample_factory_checkpoint_from_paths(checkpoints, device)
    actor_critic.load_state_dict(checkpoint["model"])
    return checkpoint


class SampleFactoryIntegrationModule:
    """Adapt a training-side integration module to the policy runner interface."""

    def __init__(self, module: Any, register_attr: str):
        self.module = module
        self._register_attr = register_attr

    def register_components(self) -> None:
        getattr(self.module, self._register_attr)()

    def build_sf_cfg_from_config(self, config: dict[str, Any]):
        return self.module.SAMPLE_FACTORY_CONFIG_PARSER.parse_eval(
            self.module.build_cli_args_from_config(config)
        )

    def make_reference_env(self, cfg: Any):
        return self.module.make_reference_env(cfg)
