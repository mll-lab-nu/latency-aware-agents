"""Sample Factory checkpoint loading and device helpers shared across workflows."""

from __future__ import annotations

from typing import Any
from pathlib import Path

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
) -> tuple[dict[str, Any], Path]:
    from training.common.checkpoints import (
        load_sample_factory_checkpoint_from_paths,
        trusted_torch_load,
    )

    if checkpoint_path is not None:
        resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = trusted_torch_load(resolved_checkpoint_path, device)
    else:
        from sample_factory.algo.learning.learner import Learner

        name_prefix = {"latest": "checkpoint", "best": "best"}[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(
            Learner.checkpoint_dir(cfg, cfg.policy_index),
            f"{name_prefix}_*",
        )
        checkpoint, resolved_checkpoint_path = load_sample_factory_checkpoint_from_paths(
            checkpoints,
            device,
        )
    actor_critic.load_state_dict(checkpoint["model"])
    return checkpoint, resolved_checkpoint_path
