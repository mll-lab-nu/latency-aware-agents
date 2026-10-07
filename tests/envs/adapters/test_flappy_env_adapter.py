from __future__ import annotations

import gymnasium
import numpy as np
import pytest

from latency_bench.core.types import Action
from latency_bench.envs.adapters.flappy import FlappyBirdEnvAdapter
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


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


def test_flappy_raw_rgb_export_preserves_unresized_render_stack(monkeypatch):
    class FakeBaseEnv:
        action_space = gymnasium.spaces.Discrete(2)
        observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

        def __init__(self):
            self.render_value = 10

        def reset(self, seed=None):
            del seed
            self.render_value = 10
            return np.zeros((12,), dtype=np.float32), {"score": 0}

        def step(self, action):
            del action
            self.render_value = 20
            return np.ones((12,), dtype=np.float32), 0.0, False, False, {"score": 1}

        def render(self):
            return np.full((4, 5, 3), self.render_value, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr("latency_bench.envs.adapters.flappy.gymnasium.make", lambda *args, **kwargs: FakeBaseEnv())

    env = FlappyBirdEnvAdapter(
        obs_resize=(2, 2),
        frame_stack=2,
        export_env_raw_rgb_frames=True,
        noop_action=Action(value=0, name="noop", is_noop=True),
    )
    try:
        reset_observation = env.reset(seed=0)
        reset_stack = reset_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]
        env.step(Action(value=0, name="noop", is_noop=True))
        policy_observation = env.observe()
        policy_stack = policy_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]

        assert reset_observation.data.shape == (2, 2, 3)
        assert reset_stack.shape == (2, 4, 5, 3)
        assert reset_stack[:, 0, 0, 0].tolist() == [10, 10]
        assert policy_observation.data.shape == (2, 2, 6)
        assert policy_stack.shape == (2, 4, 5, 3)
        assert policy_stack[:, 0, 0, 0].tolist() == [10, 20]
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
            export_env_raw_rgb_frames=False,
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
            captured["export_env_raw_rgb_frames"] = export_env_raw_rgb_frames
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

    monkeypatch.setattr("latency_bench.envs.adapters.flappy.gymnasium.make", fake_make)
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
        assert captured["export_env_raw_rgb_frames"] is False
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


def test_flappy_render_game_frame_returns_unprocessed_game_rgb(monkeypatch):
    class FakeBaseEnv:
        action_space = gymnasium.spaces.Discrete(2)
        observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

        def reset(self, seed=None):
            del seed
            return np.zeros((12,), dtype=np.float32), {"score": 0}

        def step(self, action):
            del action
            return np.ones((12,), dtype=np.float32), 0.0, False, False, {"score": 1}

        def render(self):
            return np.full((10, 20, 3), 17, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr("latency_bench.envs.adapters.flappy.gymnasium.make", lambda *args, **kwargs: FakeBaseEnv())

    env = FlappyBirdEnvAdapter(
        obs_resize=(4, 4),
        noop_action=Action(value=0, name="noop", is_noop=True),
    )
    try:
        observation = env.reset(seed=0)
        game_frame = env.render_game_frame()

        assert observation.data.shape == (4, 4, 3)
        assert game_frame.shape == (10, 20, 3)
        assert game_frame.dtype == np.uint8
        assert int(game_frame[0, 0, 0]) == 17
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
    from latency_bench.envs.adapters import flappy as flappy_module

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


def test_flappy_cpu_observe_stacks_raw_frames_between_policy_observations():
    env = FlappyBirdEnvAdapter(
        gym_id="FlappyBird-v0",
        env_fps=30,
        noop_action=Action(value=0, name="noop", is_noop=True),
        frame_stack=2,
    )
    try:
        reset_observation = env.reset(seed=0)
        first_policy_observation = env.observe()
        first_step = env.step(Action(value=0, name="noop", is_noop=True))
        second_step = env.step(Action(value=0, name="noop", is_noop=True))
        second_policy_observation = env.observe()

        assert reset_observation.data.shape[-1] == 3
        assert first_step.observation is not None
        assert second_step.observation is not None
        assert second_step.observation.data.shape[-1] == 3
        assert first_policy_observation.data.shape[-1] == 6
        assert second_policy_observation.data.shape[-1] == 6
        assert np.array_equal(second_policy_observation.data[..., :3], first_step.observation.data)
        assert np.array_equal(second_policy_observation.data[..., 3:], second_step.observation.data)
    finally:
        env.close()


def test_flappy_gpu_observe_stacks_only_policy_observed_frames(monkeypatch):
    from latency_bench.envs.adapters import flappy as flappy_module

    class FakeBaseEnv:
        action_space = gymnasium.spaces.Discrete(2)
        observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

        def close(self):
            pass

    class FakeGpuRenderWrapper:
        action_space = gymnasium.spaces.Discrete(2)
        observation_space = gymnasium.spaces.Box(0, 255, shape=(3, 2, 2), dtype=np.uint8)

        def __init__(self, *_args, **_kwargs):
            self.frame_value = 1

        def reset(self, seed=None):
            self.frame_value = 1
            return {"obs": np.full((1, 2, 2, 3), self.frame_value, dtype=np.uint8)}, [{"score": 0}]

        def step(self, action):
            self.frame_value += 1
            return (
                {"obs": np.full((1, 2, 2, 3), self.frame_value, dtype=np.uint8)},
                np.asarray([0.0], dtype=np.float32),
                np.asarray([False], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [{"score": self.frame_value}],
            )

        def close(self):
            pass

    monkeypatch.setattr(flappy_module.gymnasium, "make", lambda *args, **kwargs: FakeBaseEnv())
    monkeypatch.setattr("training.flappy_sf.gpu_rendering.FlappyGpuRenderedObsWrapper", FakeGpuRenderWrapper)

    env = FlappyBirdEnvAdapter(
        use_gpu_render=True,
        gpu_render_device="cpu",
        obs_resize=(2, 2),
        frame_stack=2,
        noop_action=Action(value=0, name="noop", is_noop=True),
    )
    try:
        reset_observation = env.reset(seed=0)
        env.observe()
        env.step(Action(value=0, name="noop", is_noop=True))
        second_step = env.step(Action(value=0, name="noop", is_noop=True))
        second_policy_observation = env.observe()

        assert second_step.observation is not None
        assert np.all(reset_observation.data == 1)
        assert np.all(second_step.observation.data == 3)
        assert np.all(second_policy_observation.data[..., :3] == 1)
        assert np.all(second_policy_observation.data[..., 3:] == 3)
    finally:
        env.close()


def test_flappy_gpu_raw_rgb_export_reads_wrapper_info(monkeypatch):
    from latency_bench.envs.adapters import flappy as flappy_module

    class FakeBaseEnv:
        action_space = gymnasium.spaces.Discrete(2)
        observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)

        def close(self):
            pass

    class FakeGpuRenderWrapper:
        action_space = gymnasium.spaces.Discrete(2)
        observation_space = gymnasium.spaces.Box(0, 255, shape=(3, 2, 2), dtype=np.uint8)

        def __init__(self, *_args, **_kwargs):
            self.frame_value = 1

        def reset(self, seed=None):
            del seed
            self.frame_value = 1
            return (
                {"obs": np.full((1, 2, 2, 3), self.frame_value, dtype=np.uint8)},
                [{ENV_RAW_RGB_FRAME_STACK_INFO_KEY: np.full((1, 5, 6, 3), 11, dtype=np.uint8)}],
            )

        def step(self, action):
            del action
            self.frame_value = 2
            return (
                {"obs": np.full((1, 2, 2, 3), self.frame_value, dtype=np.uint8)},
                np.asarray([0.0], dtype=np.float32),
                np.asarray([False], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [{ENV_RAW_RGB_FRAME_STACK_INFO_KEY: np.full((1, 5, 6, 3), 22, dtype=np.uint8)}],
            )

        def close(self):
            pass

    monkeypatch.setattr(flappy_module.gymnasium, "make", lambda *args, **kwargs: FakeBaseEnv())
    monkeypatch.setattr("training.flappy_sf.gpu_rendering.FlappyGpuRenderedObsWrapper", FakeGpuRenderWrapper)

    env = FlappyBirdEnvAdapter(
        use_gpu_render=True,
        gpu_render_device="cpu",
        obs_resize=(2, 2),
        frame_stack=2,
        export_env_raw_rgb_frames=True,
        noop_action=Action(value=0, name="noop", is_noop=True),
    )
    try:
        reset_observation = env.reset(seed=0)
        env.step(Action(value=0, name="noop", is_noop=True))
        policy_observation = env.observe()

        assert reset_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY].shape == (2, 5, 6, 3)
        assert reset_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][:, 0, 0, 0].tolist() == [11, 11]
        assert policy_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][:, 0, 0, 0].tolist() == [11, 22]
        assert policy_observation.data.shape == (2, 2, 6)
    finally:
        env.close()


def test_flappy_observe_can_stack_chw_for_sample_factory():
    env = FlappyBirdEnvAdapter(
        gym_id="FlappyBird-v0",
        env_fps=30,
        noop_action=Action(value=0, name="noop", is_noop=True),
        obs_resize=(84, 84),
        frame_stack=4,
        policy_observation_layout="chw",
    )
    try:
        reset_observation = env.reset(seed=0)
        policy_observation = env.observe()

        assert reset_observation.data.shape == (84, 84, 3)
        assert policy_observation.data.shape == (12, 84, 84)
        assert policy_observation.data.dtype == np.uint8
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


def test_flappy_raw_render_is_pure_function_of_seed_and_actions():
    # The wing-animation cycle and ground scroll used to persist across
    # reset(), so an env's rendered pixels depended on how many episodes the
    # instance had already run. That fed different frames to the policy in
    # measure vs replay runs and skewed sim-vs-real returns.
    def rollout_frames(env, seed, steps=14):
        env.reset(seed=seed)
        frames = []
        for step in range(steps):
            stack = np.asarray(env.observe().metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY])
            frames.append(stack.copy())
            action = (
                Action(value=1, name="flap", is_oneshot=True)
                if step % 5 == 0
                else Action(value=0, name="noop", is_noop=True)
            )
            env.step(action)
        return frames

    reused = FlappyBirdEnvAdapter(
        gym_id="FlappyBird-v0",
        env_fps=30,
        noop_action=Action(value=0, name="noop", is_noop=True),
        export_env_raw_rgb_frames=True,
    )
    fresh = FlappyBirdEnvAdapter(
        gym_id="FlappyBird-v0",
        env_fps=30,
        noop_action=Action(value=0, name="noop", is_noop=True),
        export_env_raw_rgb_frames=True,
    )
    try:
        first = rollout_frames(reused, seed=123)
        rollout_frames(reused, seed=456)  # interleave another episode
        second = rollout_frames(reused, seed=123)
        baseline = rollout_frames(fresh, seed=123)

        for step, (a, b, c) in enumerate(zip(first, second, baseline)):
            np.testing.assert_array_equal(a, b, err_msg=f"reset leaked render state at step {step}")
            np.testing.assert_array_equal(a, c, err_msg=f"reused env differs from fresh env at step {step}")
    finally:
        reused.close()
        fresh.close()
