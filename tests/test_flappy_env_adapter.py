from __future__ import annotations

import gymnasium
import numpy as np
import pytest

from latency_bench.core.types import Action
from latency_bench.envs.flappy import FlappyBirdEnvAdapter


def _make_adapter() -> FlappyBirdEnvAdapter:
    return FlappyBirdEnvAdapter(
        gym_id="FlappyBird-v0",
        env_fps=30,
        noop_action=Action(value=0, name="noop", is_noop=True),
    )


def test_flappy_reset_returns_rgb_image_observation():
    env = _make_adapter()
    try:
        observation = env.reset(seed=0)

        assert observation.env_step == 0
        assert observation.sim_time_ms == 0.0
        assert observation.data.ndim == 3
        assert observation.data.shape[2] == 3
        assert observation.metadata["score"] == 0
        assert observation.metadata["numeric_obs"].shape == (12,)
    finally:
        env.close()


def test_flappy_step_advances_one_frame_and_records_action_info():
    env = _make_adapter()
    try:
        env.reset(seed=0)
        result = env.step(Action(value=0, name="noop", is_noop=True))

        assert result.observation is not None
        assert result.observation.env_step == 1
        assert result.observation.data.ndim == 3
        assert result.reward == 0.1
        assert not result.done
        assert not result.truncated
        assert result.info["env_step"] == 1
        assert result.info["applied_action"] == 0
        assert result.info["applied_action_name"] == "noop"
        assert "score" in result.info
    finally:
        env.close()


def test_flappy_gpu_backend_uses_explicit_renderer_args(monkeypatch):
    captured = {}

    class FakeBaseEnv:
        action_space = gymnasium.spaces.Discrete(2)
        observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

        def close(self):
            captured["base_closed"] = True

    class FakeGpuRenderWrapper:
        action_space = gymnasium.spaces.Discrete(2)
        observation_space = gymnasium.spaces.Box(0, 255, shape=(3, 84, 84), dtype=np.uint8)

        def __init__(
            self,
            env,
            *,
            obs_resize,
            frame_stack,
            gpu_render_device,
            gpu_render_profile=False,
            gpu_render_profile_interval=1000,
            worker_index=0,
            vector_index=0,
            env_id=0,
        ):
            captured["base_env"] = env
            captured["obs_resize"] = obs_resize
            captured["wrapper_frame_stack"] = frame_stack
            captured["gpu_render_device"] = gpu_render_device
            captured["gpu_render_profile"] = gpu_render_profile
            captured["gpu_render_profile_interval"] = gpu_render_profile_interval
            captured["worker_index"] = worker_index
            captured["vector_index"] = vector_index
            captured["env_id"] = env_id

        def reset(self, seed=None):
            captured["reset_seed"] = seed
            return {"obs": np.zeros((1, 3, 84, 84), dtype=np.uint8)}, [{"score": 0}]

        def step(self, action):
            captured["step_action"] = int(action)
            return (
                {"obs": np.full((1, 3, 84, 84), 7, dtype=np.uint8)},
                np.asarray([1.25], dtype=np.float32),
                np.asarray([False], dtype=np.bool_),
                np.asarray([True], dtype=np.bool_),
                [{"score": 3}],
            )

        def close(self):
            captured["closed"] = True

    def fake_make(*args, **kwargs):
        captured["gym_make_args"] = args
        captured["gym_make_kwargs"] = kwargs
        return FakeBaseEnv()

    monkeypatch.setattr("latency_bench.envs.flappy.gymnasium.make", fake_make)
    monkeypatch.setattr("training.flappy_sf.gpu_rendering.FlappyGpuRenderedObsWrapper", FakeGpuRenderWrapper)
    noop = Action(value=0, name="noop", is_noop=True)

    env = FlappyBirdEnvAdapter(
        use_gpu_render=True,
        gpu_render_device="cpu",
        gpu_render_profile=True,
        gpu_render_profile_interval=17,
        obs_resize=(84, 84),
        frame_stack=4,
        noop_action=noop,
    )
    try:
        observation = env.reset(seed=123)
        assert captured["reset_seed"] == 123
        assert observation.data.shape == (84, 84, 3)
        assert observation.data.dtype == np.uint8
        assert observation.env_step == 0
        assert observation.metadata["score"] == 0
        assert env.observe().data.shape == (84, 84, 12)

        result = env.step(Action(value=1, name="flap", is_oneshot=True))
        assert captured["step_action"] == 1
        assert result.observation is not None
        assert result.observation.data.shape == (84, 84, 3)
        assert result.reward == float(np.float32(1.25))
        assert result.done is False
        assert result.truncated is True
        assert result.info["applied_action"] == 1
        assert result.info["applied_action_name"] == "flap"
        assert captured["gym_make_kwargs"]["render_mode"] is None
        assert captured["obs_resize"] == (84, 84)
        assert captured["wrapper_frame_stack"] == 1
        assert captured["gpu_render_device"] == "cpu"
        assert captured["gpu_render_profile"] is True
        assert captured["gpu_render_profile_interval"] == 17
    finally:
        env.close()

    assert captured["closed"] is True


def test_flappy_render_frame_returns_latest_rgb_frame():
    env = _make_adapter()
    try:
        observation = env.reset(seed=0)
        frame = env.render_frame()

        assert frame.shape == observation.data.shape
        assert frame.shape[2] == 3
    finally:
        env.close()


def test_flappy_resize_keeps_hwc_rgb_layout():
    env = FlappyBirdEnvAdapter(
        gym_id="FlappyBird-v0",
        env_fps=30,
        noop_action=Action(value=0, name="noop", is_noop=True),
        obs_resize=(84, 84),
    )
    try:
        observation = env.reset(seed=0)

        assert observation.data.shape == (84, 84, 3)
        assert observation.data.dtype == np.uint8
    finally:
        env.close()


def test_flappy_chw_render_is_converted_to_hwc_rgb(monkeypatch):
    from latency_bench.envs import flappy as flappy_module

    class FakeGymEnv:
        def reset(self, seed=None):
            return np.zeros((12,), dtype=np.float64), {"score": 0}

        def render(self):
            return np.zeros((3, 10, 20), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(flappy_module.gymnasium, "make", lambda *args, **kwargs: FakeGymEnv())

    env = FlappyBirdEnvAdapter(env_fps=30, noop_action=Action(value=0, name="noop", is_noop=True))
    try:
        observation = env.reset(seed=0)

        assert observation.data.shape == (10, 20, 3)
        assert observation.data.dtype == np.uint8
    finally:
        env.close()


def test_flappy_observe_stacks_only_observed_frames():
    env = FlappyBirdEnvAdapter(
        gym_id="FlappyBird-v0",
        env_fps=30,
        noop_action=Action(value=0, name="noop", is_noop=True),
        frame_stack=2,
    )
    try:
        reset_observation = env.reset(seed=0)
        first_policy_observation = env.observe()
        env.step(Action(value=0, name="noop", is_noop=True))
        second_step = env.step(Action(value=0, name="noop", is_noop=True))
        second_policy_observation = env.observe()

        assert reset_observation.data.shape[-1] == 3
        assert second_step.observation is not None
        assert second_step.observation.data.shape[-1] == 3
        assert first_policy_observation.data.shape[-1] == 6
        assert second_policy_observation.data.shape[-1] == 6
        assert np.array_equal(second_policy_observation.data[..., :3], reset_observation.data)
        assert np.array_equal(second_policy_observation.data[..., 3:], second_step.observation.data)
    finally:
        env.close()


def test_flappy_gpu_backend_reset_and_step_contract():
    pytest.importorskip("torch")
    env = FlappyBirdEnvAdapter(
        use_gpu_render=True,
        gpu_render_device="cpu",
        obs_resize=(84, 84),
        frame_stack=4,
        noop_action=Action(value=0, name="noop", is_noop=True),
    )
    try:
        observation = env.reset(seed=123)

        assert observation.env_step == 0
        assert observation.data.shape == (84, 84, 3)
        assert observation.data.dtype == np.uint8
        assert "numeric_obs" in observation.metadata
        assert env.observe().data.shape == (84, 84, 12)

        result = env.step(Action(value=0, name="noop", is_noop=True))

        assert result.observation is not None
        assert result.observation.env_step == 1
        assert result.observation.data.shape == (84, 84, 3)
        assert result.observation.data.dtype == np.uint8
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.truncated, bool)
        assert result.info["applied_action"] == 0
        assert result.info["applied_action_name"] == "noop"
        assert "numeric_obs" not in result.info
    finally:
        env.close()
