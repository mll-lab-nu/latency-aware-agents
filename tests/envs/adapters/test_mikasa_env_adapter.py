from __future__ import annotations

import numpy as np
import torch

from latency_bench.core.types import Action
from latency_bench.envs.adapters.mikasa import MikasaEnvStepBackend
from latency_bench.envs.tasks.mikasa import (
    MIKASA_IMAGE_VIEWS_INFO_KEY,
    MIKASA_STATE_INFO_KEY,
)


def test_latency_backend_exposes_two_views_and_direct_action():
    class FakeProfileEnv:
        def __init__(self):
            self.actions = []
            self.closed = False
            self.step_index = 0
            self.unwrapped = self
            self.device = "cpu"

        def reset(self, *, seed, options=None):
            return rgb_to_observation(np.zeros((1, 2, 3, 6), dtype=np.uint8)), {}

        def step(self, action):
            self.actions.append(action)
            rgb = np.zeros((1, 2, 3, 6), dtype=np.uint8)
            rgb[..., :3] = 11
            rgb[..., 3:] = 22
            success = self.step_index == 0
            self.step_index += 1
            return rgb_to_observation(rgb), np.asarray([1.5]), np.asarray([False]), np.asarray([False]), {
                "success": np.asarray([success])
            }

        def close(self):
            self.closed = True

    def rgb_to_observation(rgb):
        return {"rgb": rgb, "proprio": torch.arange(7).reshape(1, 7)}

    raw_env = FakeProfileEnv()
    env = MikasaEnvStepBackend(
        config={
            "env": {
                "env_fps": 20,
                "noop_action": [0.0] * 7,
            }
        },
        num_slots=1,
        env=raw_env,
    )
    reset_observation = env.reset_slot(0, episode_id=0, seed=123)
    first_result = env.step_slots({0: Action(value=[0.1] * 7)})[0].result
    result = env.step_slots({0: Action(value=[0.1] * 7)})[0].result

    assert reset_observation.metadata[MIKASA_IMAGE_VIEWS_INFO_KEY].shape == (2, 2, 3, 3)
    np.testing.assert_array_equal(
        reset_observation.metadata[MIKASA_STATE_INFO_KEY], np.arange(7)
    )
    assert [action.shape for action in raw_env.actions] == [(1, 7), (1, 7)]
    np.testing.assert_allclose(raw_env.actions[0], [[0.1] * 7])
    np.testing.assert_allclose(result.info["applied_action"], [0.1] * 7)
    observation = env.observe_slots([0])[0]
    assert [view.mean() for view in observation.metadata[MIKASA_IMAGE_VIEWS_INFO_KEY]] == [
        11,
        22,
    ]
    assert result.reward == 1.5
    assert first_result.info["success"] is True
    assert result.info["success"] is False
    assert result.info["task_metrics"] == {"success": 1.0}
    assert result.info["applied_action_name"] is None
