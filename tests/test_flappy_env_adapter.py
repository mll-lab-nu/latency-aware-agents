from __future__ import annotations

import numpy as np

from latency_bench.core.types import Action
from latency_bench.envs.flappy import FlappyBirdEnvAdapter, FlappySampleFactoryGpuEnvAdapter


def _make_adapter() -> FlappyBirdEnvAdapter:
    return FlappyBirdEnvAdapter(
        gym_id="FlappyBird-v0",
        fps_env=30,
        noop_action=Action(value=0, name="noop", is_noop=True),
        render_mode="rgb_array",
        observation_mode="image",
        use_lidar=False,
        normalize_obs=True,
        audio_on=False,
    )


def _gpu_config(obs_mode: str = "image_only") -> dict:
    return {
        "experiment": {"seed": 11},
        "env": {
            "name": "flappy",
            "fps_env": 30,
            "gym_id": "FlappyBird-v0",
            "use_lidar": False,
            "normalize_obs": True,
            "audio_on": False,
            "pipe_gap": 100,
            "bird_color": "yellow",
            "pipe_color": "green",
            "background": "day",
            "debug": False,
            "score_limit": -1,
            "obs_resize": [84, 84],
            "obs_mode": obs_mode,
            "frame_stack": 4,
            "use_gpu_render": True,
            "gpu_render_device": "cpu",
        },
    }


def test_flappy_reset_returns_rgb_image_observation():
    env = _make_adapter()
    try:
        observation = env.reset(seed=0)

        assert observation.env_step == 0
        assert observation.sim_time_ms == 0.0
        assert observation.data.ndim == 3
        assert observation.data.shape[2] == 3
        assert observation.metadata["score"] == 0
        assert observation.metadata["observation_mode"] == "image"
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


def test_flappy_sample_factory_gpu_adapter_uses_renderer_without_internal_latency(monkeypatch):
    from training.flappy_sf import integration

    captured = {}

    class FakeSampleFactoryGpuEnv:
        def reset(self, **kwargs):
            captured["reset_seed"] = kwargs.get("seed")
            return {"obs": np.zeros((1, 12, 84, 84), dtype=np.uint8)}, [{"score": 0}]

        def step(self, action):
            captured["step_action"] = action
            return (
                {"obs": np.ones((1, 12, 84, 84), dtype=np.uint8)},
                np.asarray([0.1], dtype=np.float32),
                np.asarray([False], dtype=np.bool_),
                np.asarray([False], dtype=np.bool_),
                [{"score": 0}],
            )

        def close(self):
            captured["closed"] = True

    def fake_make_flappy_sf_env(_full_env_name, cfg=None, env_config=None, render_mode=None):
        captured["cfg"] = cfg
        captured["env_config"] = env_config
        captured["render_mode"] = render_mode
        return FakeSampleFactoryGpuEnv()

    monkeypatch.setattr(integration, "make_flappy_sf_env", fake_make_flappy_sf_env)
    noop = Action(value=0, name="noop", is_noop=True)

    env = FlappySampleFactoryGpuEnvAdapter(config=_gpu_config(obs_mode="latency_channel"), noop_action=noop)
    try:
        assert captured["cfg"].use_gpu_render is True
        assert captured["cfg"].obs_mode == "image_only"
        assert captured["cfg"].queue_latency_values == "0"
        assert captured["cfg"].gpu_render_batch_size == 1
        assert captured["env_config"].env_id == 0

        observation = env.reset(seed=123)
        assert captured["reset_seed"] == 123
        assert observation.data.shape == (12, 84, 84)
        assert observation.data.dtype == np.uint8
        assert observation.env_step == 0
        assert observation.metadata["score"] == 0

        result = env.step(Action(value=1, name="flap", is_oneshot=True))
        assert captured["step_action"] == 1
        assert result.observation is not None
        assert result.observation.data.shape == (12, 84, 84)
        assert result.reward == float(np.float32(0.1))
        assert result.done is False
        assert result.truncated is False
        assert result.info["applied_action"] == 1
        assert result.info["applied_action_name"] == "flap"
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
