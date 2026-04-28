from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """
    CleanRL CNN agent identical to the one in cleanrl_ppo_demon_attack.py.
    Expects obs shape (C, H, W) = (4, 84, 84) with H=W=84.
    """

    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int):
        super().__init__()
        c, h, w = obs_shape
        assert h == 84 and w == 84, f"Expected 84x84 obs; got {obs_shape}"

        self.network = nn.Sequential(
            _layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            _layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = _layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = _layer_init(nn.Linear(512, 1), std=1.0)

    @torch.no_grad()
    def get_deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) uint8 — returns (B,) int64 action tensor."""
        hidden = self.network(x.float() / 255.0)
        return torch.argmax(self.actor(hidden), dim=-1)


def load_cleanrl(
    checkpoint: str,
    obs_shape: Tuple[int, int, int],
    num_actions: int,
    device: str = "cpu",
) -> Agent:
    """
    Load a CleanRL PPO checkpoint (.pt) saved by save_checkpoint() in the training script.
    The checkpoint dict has key 'model' with the agent state_dict.
    """
    agent = Agent(obs_shape=obs_shape, num_actions=num_actions)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt)  # fallback if someone saved the state_dict directly
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent
