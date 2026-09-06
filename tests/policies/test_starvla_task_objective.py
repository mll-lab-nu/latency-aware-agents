"""Semantic checks for the frozen H1 actor objectives."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
PARKOUR_ACTOR = (
    ROOT
    / "outputs/starvla/artifacts/models/zero-latency/extreme-parkour-go2"
    / "small-policy/ppo-v1/model_14999.pt"
)
HAIC_ACTOR = Path(
    os.environ.get(
        "STARVLA_HAIC_ACTOR_CHECKPOINT",
        ROOT
        / "outputs/starvla/artifacts/models/zero-latency/haic-pull-cart"
        / "small-policy/ppo-v1/actor_adapt.pt",
    )
)
PARKOUR_OUTPUT_MIN = [-1.0] * 32 + [-np.pi / 1.5] * 2
PARKOUR_OUTPUT_MAX = [1.0] * 32 + [np.pi / 1.5] * 2


def _objective(
    name: str,
    actor_checkpoint: Path,
    output_min: list[float],
    output_max: list[float],
) -> object:
    from latency_bench.policy.starvla_task_objective import TaskActionObjective

    return TaskActionObjective(
        {
            "name": name,
            "actor_checkpoint": str(actor_checkpoint),
            "output_min": output_min,
            "output_max": output_max,
        }
    )


def _assert_actor_is_outside_optimizer(objective: object) -> None:
    holder = torch.nn.Module()
    holder.student = torch.nn.Linear(1, 1)
    holder.task_objective = objective
    optimizer = torch.optim.AdamW(holder.parameters(), lr=1e-3)
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(
        id(parameter) not in optimizer_parameters
        for parameter in objective.objective.actor.parameters()
    )


@pytest.mark.skipif(
    not PARKOUR_ACTOR.is_file(), reason="Parkour actor fixture is unavailable"
)
def test_parkour_objective_uses_five_fresh_observations_and_backpropagates():
    objective = _objective(
        "parkour", PARKOUR_ACTOR, PARKOUR_OUTPUT_MIN, PARKOUR_OUTPUT_MAX
    )
    observations = np.stack(
        [
            np.linspace(-1.0, 1.0, 753, dtype=np.float32) + index * 0.01
            for index in range(5)
        ]
    )
    examples = [
        {
            "actor_observation": observations,
            "action_target": np.linspace(-0.5, 0.5, 34, dtype=np.float32)[None, :],
            "termination": np.asarray([False, False, False, False, True]),
        }
    ]
    prediction = torch.linspace(-0.2, 0.2, 34).reshape(1, 1, 34).requires_grad_()
    actor_inputs: list[torch.Tensor] = []
    handle = objective.objective.actor.register_forward_hook(
        lambda _module, inputs, _output: actor_inputs.append(inputs[0].detach().clone())
    )
    try:
        loss = objective(prediction, examples)
    finally:
        handle.remove()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert len(actor_inputs) == 2
    assert actor_inputs[0].shape == (5, 753)
    assert not torch.equal(actor_inputs[0][0], actor_inputs[0][1])

    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.norm().item() > 0.0
    assert all(
        not parameter.requires_grad for parameter in objective.objective.actor.parameters()
    )
    _assert_actor_is_outside_optimizer(objective)


@pytest.mark.skipif(
    not HAIC_ACTOR.is_file(), reason="exported HAIC actor_adapt fixture is unavailable"
)
def test_haic_objective_backpropagates_through_frozen_actor():
    objective = _objective("haic", HAIC_ACTOR, [-1.0] * 256, [1.0] * 256)
    examples = [
        {
            "actor_input": np.linspace(-1.0, 1.0, 605, dtype=np.float32),
            "teacher_action": np.linspace(-0.5, 0.5, 23, dtype=np.float32),
            "termination": np.asarray([True]),
        }
    ]
    prediction = torch.linspace(-0.2, 0.2, 256).reshape(1, 1, 256).requires_grad_()

    loss = objective(prediction, examples)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.norm().item() > 0.0
    assert all(
        not parameter.requires_grad for parameter in objective.objective.actor.parameters()
    )
    _assert_actor_is_outside_optimizer(objective)
