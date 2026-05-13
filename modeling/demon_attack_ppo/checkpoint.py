from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from modeling.demon_attack_ppo.agent import DemonAttackPpoAgent
from modeling.demon_attack_ppo.observation import (
    DemonAttackPpoObservationConfig,
    observation_config_from_mapping,
)


CHECKPOINT_FORMAT = "demon_attack_ppo"


def save_demon_attack_ppo_checkpoint(
    path: str | os.PathLike[str],
    agent: DemonAttackPpoAgent,
    optimizer: Any | None,
    observation_config: DemonAttackPpoObservationConfig,
    global_step: int,
    train_config: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "agent": agent.state_dict(),
        "observation_config": observation_config.to_dict(),
        "global_step": int(global_step),
        "train_config": dict(train_config or {}),
    }
    if train_config is not None:
        checkpoint["args"] = dict(train_config)
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_demon_attack_ppo_checkpoint(
    path: str | os.PathLike[str],
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def observation_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> DemonAttackPpoObservationConfig:
    if checkpoint.get("format") == CHECKPOINT_FORMAT and "observation_config" in checkpoint:
        return DemonAttackPpoObservationConfig.from_dict(checkpoint["observation_config"])

    config = checkpoint.get("observation_config")
    if isinstance(config, Mapping):
        return DemonAttackPpoObservationConfig.from_dict(config)

    train_config = checkpoint.get("train_config")
    if isinstance(train_config, Mapping):
        return observation_config_from_mapping(train_config)

    args = checkpoint.get("args")
    if isinstance(args, Mapping):
        return observation_config_from_mapping(args)

    agent_state = checkpoint.get("agent", {})
    return _infer_observation_config_from_channels(_infer_in_channels(agent_state))


def build_agent_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: str | torch.device = "cpu",
    n_actions: int | None = None,
) -> DemonAttackPpoAgent:
    observation_config = observation_config_from_checkpoint(checkpoint)
    agent_state = checkpoint["agent"]
    if n_actions is None:
        n_actions = infer_n_actions_from_state(agent_state)
    agent = DemonAttackPpoAgent(
        in_channels=observation_config.in_channels,
        image_size=observation_config.image_size,
        n_actions=n_actions,
    ).to(device)
    agent.load_state_dict(agent_state)
    return agent


def infer_n_actions_from_state(agent_state: Mapping[str, Any]) -> int:
    actor_weight = agent_state.get("actor.weight")
    if actor_weight is None:
        return 1
    return int(actor_weight.shape[0])


def _infer_in_channels(agent_state: Mapping[str, Any]) -> int:
    for key in ("network.0.0.weight", "network.0.weight"):
        weight = agent_state.get(key)
        if weight is not None:
            return int(weight.shape[1])
    return 5


def _infer_observation_config_from_channels(
    in_channels: int,
) -> DemonAttackPpoObservationConfig:
    preferred_stack_frames = (4, 1, 2, 3, 5, 6, 8)
    for stack_frames in preferred_stack_frames:
        for base_channels in (1, 3):
            for has_latency in (False, True):
                expected = base_channels * stack_frames + int(has_latency)
                if expected != in_channels:
                    continue
                return DemonAttackPpoObservationConfig(
                    image_size=84,
                    base_channels=base_channels,
                    stack_frames=stack_frames,
                    obs_mode="latency_channel" if has_latency else "image_only",
                    frame_skip_values=(0, 1, 2, 3, 4, 5),
                )
    return DemonAttackPpoObservationConfig(
        image_size=84,
        base_channels=1,
        stack_frames=max(1, in_channels),
        obs_mode="image_only",
        frame_skip_values=(0, 1, 2, 3, 4, 5),
    )
