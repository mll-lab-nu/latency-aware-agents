from __future__ import annotations

import numpy as np

from latency_bench.core.types import Observation
from latency_bench.envs.base import DummyEnvAdapter
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer


def _observation(data, env_step: int) -> Observation:
    return Observation(data=data, env_step=env_step, sim_time_ms=float(env_step), metadata={"env_step": env_step})


def test_observation_builder_stacks_hwc_frames_with_padding():
    history = HistoryBuffer(max_len=4)
    for value, env_step in [(10, 0), (20, 1), (30, 2)]:
        history.append(_observation(np.full((2, 3, 1), value, dtype=np.uint8), env_step))

    observation = ObservationBuilder(frame_stack=4).build(
        DummyEnvAdapter(),
        history,
    )

    assert observation.env_step == 2
    assert observation.data.shape == (2, 3, 4)
    assert observation.data[0, 0, :].tolist() == [10, 10, 20, 30]


def test_observation_builder_stacks_chw_frames_on_channel_axis():
    history = HistoryBuffer(max_len=2)
    history.append(_observation(np.full((3, 2, 2), 1, dtype=np.uint8), 0))
    history.append(_observation(np.full((3, 2, 2), 2, dtype=np.uint8), 1))

    observation = ObservationBuilder(frame_stack=2).build(
        DummyEnvAdapter(),
        history,
    )

    assert observation.data.shape == (6, 2, 2)
    assert observation.data[:, 0, 0].tolist() == [1, 1, 1, 2, 2, 2]


def test_observation_builder_leaves_non_image_observation_unstacked():
    latest = _observation({"state": 1}, 1)
    history = HistoryBuffer(max_len=4)
    history.append(latest)

    observation = ObservationBuilder(frame_stack=4).build(
        DummyEnvAdapter(),
        history,
    )

    assert observation is latest


def test_dummy_env_observe_returns_current_state_without_stepping():
    env = DummyEnvAdapter(env_fps=30, max_steps=10)
    env.reset(seed=123)
    env.step(env.noop_action)
    env.step(env.noop_action)

    observation = env.observe()

    assert env.env_step == 2
    assert observation.env_step == 2
    assert observation.sim_time_ms == env.frame_ms * 2
    assert observation.data == {"state": 2}
