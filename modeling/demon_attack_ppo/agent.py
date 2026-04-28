from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(
    layer: nn.Module,
    std: float = float(np.sqrt(2)),
    bias_const: float = 0.0,
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DemonAttackPpoAgent(nn.Module):
    """CleanRL-style CNN actor-critic used by the DemonAttack PPO trainer."""

    def __init__(self, in_channels: int, image_size: int, n_actions: int):
        super().__init__()
        if int(n_actions) <= 0:
            raise ValueError("n_actions must be positive")
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, image_size, image_size)
            flatten_dim = int(self.network(sample).shape[1])
        self.network = nn.Sequential(
            self.network,
            layer_init(nn.Linear(flatten_dim, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, int(n_actions)), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def _hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.float() / 255.0)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self._hidden(x))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self._hidden(x))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self._hidden(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
