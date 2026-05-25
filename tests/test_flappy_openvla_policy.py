from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from latency_bench.core.types import Action, Observation
from latency_bench.policy.flappy_openvla import FlappyOpenVlaPolicyRunner, OpenVlaPolicyConfig
from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


class _RecordingOpenVlaModel:
    def __init__(self):
        self.env_obs = None

    def predict_action_batch(self, **kwargs):
        self.env_obs = kwargs["env_obs"]
        return np.asarray([[0.0, 1.0]], dtype=np.float32), {"forward_inputs": {"action_tokens": np.asarray([1, 2])}}


def _policy_config() -> OpenVlaPolicyConfig:
    return OpenVlaPolicyConfig(
        base_model_path=Path("/missing/base"),
        checkpoint_path=Path("/missing/checkpoint"),
        prompt="play flappy",
        image_size=4,
        action_dim=2,
        num_action_chunks=1,
        add_value_head=False,
        max_prompt_length=50,
        num_images_in_input=1,
        center_crop=True,
        trust_remote_code=True,
        unnorm_key="flappy_bird",
        device=torch.device("cpu"),
    )


def test_flappy_openvla_prefers_raw_rgb_frame_stack_metadata():
    model = _RecordingOpenVlaModel()
    policy = FlappyOpenVlaPolicyRunner(
        model=model,
        config=_policy_config(),
        device=torch.device("cpu"),
        action_by_raw_id={
            0: Action(value=0, name="noop", is_noop=True),
            1: Action(value=1, name="flap", is_oneshot=True),
        },
    )
    processed_frame = np.zeros((84, 84, 3), dtype=np.uint8)
    raw_stack = np.zeros((2, 5, 6, 3), dtype=np.uint8)
    raw_stack[-1] = 77

    output = policy.predict(
        Observation(
            data=processed_frame,
            env_step=0,
            sim_time_ms=0.0,
            metadata={ENV_RAW_RGB_FRAME_STACK_INFO_KEY: raw_stack},
        )
    )

    assert output.action.value == 1
    assert model.env_obs is not None
    input_frame = model.env_obs["main_images"].detach().cpu().numpy()[0]
    assert input_frame.shape == (4, 4, 3)
    assert input_frame.mean() == 77
    assert output.metadata["input_frame_shape"] == [4, 4, 3]


def test_flappy_openvla_requires_raw_rgb_frame_stack_metadata():
    policy = FlappyOpenVlaPolicyRunner(
        model=_RecordingOpenVlaModel(),
        config=_policy_config(),
        device=torch.device("cpu"),
        action_by_raw_id={
            0: Action(value=0, name="noop", is_noop=True),
            1: Action(value=1, name="flap", is_oneshot=True),
        },
    )

    with pytest.raises(KeyError, match=ENV_RAW_RGB_FRAME_STACK_INFO_KEY):
        policy.predict(Observation(data=np.zeros((84, 84, 3), dtype=np.uint8), env_step=0, sim_time_ms=0.0))
