from __future__ import annotations

from stable_baselines3 import PPO


def load_sb3(checkpoint: str) -> PPO:
    """Load an SB3 PPO checkpoint (.zip) onto CPU."""
    return PPO.load(checkpoint, device="cpu")
