from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.framework.WM4A.WanOFT import Wan_OFT


@pytest.mark.parametrize("loss_type", ["mse", "l2"])
def test_qwen_oft_mse_loss_uses_only_active_action_dimensions(loss_type: str) -> None:
    model = Qwenvl_OFT.__new__(Qwenvl_OFT)
    torch.nn.Module.__init__(model)
    model.action_env_dim = 3
    prediction = torch.tensor([[[0.5, -0.5, 0.25, 99.0]]])
    target = torch.tensor([[[1.0, -1.0, 0.0, -99.0]]])

    loss = model._compute_action_loss_for_type(
        prediction,
        target,
        loss_type=loss_type,
    )

    assert loss == pytest.approx(F.mse_loss(prediction[..., :3], target[..., :3]).item())


def test_qwen_oft_continuous_projector_bypasses_text_bins_and_conditions_queries() -> None:
    model = Qwenvl_OFT.__new__(Qwenvl_OFT)
    nn.Module.__init__(model)
    model.state_encoding = "continuous_projector"
    model.action_model = nn.Module()
    model.action_model.state_projector = nn.Linear(2, 3, bias=False)
    model.action_model.state_projector.weight.data.copy_(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    )
    assert "action_model.state_projector.weight" in dict(model.named_parameters())
    instructions = ["move forward"]
    state = [np.array([[0.25, -0.5]], dtype=np.float32)]

    assert model._state_conditioned_instructions(instructions, state) == instructions

    conditioned = model._condition_action_queries(torch.zeros(1, 2, 3), state)

    assert torch.equal(
        conditioned,
        torch.tensor([[[0.25, -0.5, 0.75], [0.25, -0.5, 0.75]]]),
    )


def test_qwen_oft_discretized_text_path_remains_the_default() -> None:
    model = Qwenvl_OFT.__new__(Qwenvl_OFT)
    nn.Module.__init__(model)
    model.state_encoding = "discretized_text"
    state = [np.array([[0.0, 1.0]], dtype=np.float32)]

    assert model._state_conditioned_instructions(["move forward"], state) == [
        "move forward [STATE] 128 255 [ACTION]"
    ]


def test_pi_v3_continuous_state_path_keeps_state_out_of_instruction() -> None:
    model = Qwen_PI_v3.__new__(Qwen_PI_v3)
    model.state_encoding = "continuous_projector"
    state = [np.array([[0.25, -0.5]], dtype=np.float32)]

    assert model._state_conditioned_instructions(["move forward"], state) == ["move forward"]


def test_wan_oft_continuous_projector_conditions_all_action_queries() -> None:
    model = Wan_OFT.__new__(Wan_OFT)
    nn.Module.__init__(model)
    model.state_encoding = "continuous_projector"
    model.action_model = nn.Module()
    model.action_model.state_projector = nn.Linear(2, 3, bias=False)
    model.action_model.state_projector.weight.data.copy_(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    )
    state = [np.array([[0.25, -0.5]], dtype=np.float32)]

    assert model._state_conditioned_instructions(["move forward"], state) == ["move forward"]
    conditioned = model._condition_action_queries(torch.zeros(1, 2, 3), state)

    assert torch.equal(
        conditioned,
        torch.tensor([[[0.25, -0.5, 0.75], [0.25, -0.5, 0.75]]]),
    )
