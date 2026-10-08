"""Shared prefix conditioning for action-chunk flow matching heads."""

from __future__ import annotations

import torch


def flow_training_inputs(
    actions: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    action_prefix_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a clean committed prefix and a normally noised action suffix."""
    action_time = time[:, None].expand(actions.shape[:2])
    if action_prefix_mask is not None:
        action_time = torch.where(action_prefix_mask, torch.ones_like(action_time), action_time)
    trajectory = action_time[..., None] * actions + (1 - action_time[..., None]) * noise
    return trajectory, action_time


def clamp_action_prefix(
    actions: torch.Tensor,
    action_prefix: torch.Tensor | None,
    action_prefix_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return an action trajectory whose committed prefix is exact."""
    if action_prefix is None:
        return actions
    return torch.where(action_prefix_mask[..., None], action_prefix, actions)


def masked_action_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Average equally per request over action heads carrying teacher labels."""
    head_loss = ((prediction - target) ** 2).mean(dim=-1)
    if action_loss_mask is None:
        return head_loss.mean()
    return ((head_loss * action_loss_mask).sum(-1) / action_loss_mask.sum(-1)).mean()
