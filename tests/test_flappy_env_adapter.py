from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

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


def _flappy_sf_gpu_config(obs_mode: str = "image_only") -> dict:
    return {
        "experiment": {
            "name": "flappy_gpu_adapter_test",
            "seed": 0,
            "num_episodes": 1,
        },
        "env": {
            "name": "flappy",
            "gym_id": "FlappyBird-v0",
            "fps_env": 30,
            "use_lidar": False,
            "normalize_obs": True,
            "audio_on": False,
            "frame_stack": 4,
            "obs_resize": [84, 84],
            "obs_mode": obs_mode,
            "use_gpu_render": True,
            "gpu_render_device": "cpu",
            "gpu_render_profile": False,
            "gpu_render_profile_interval": 1000,
            "pipe_gap": 100,
            "bird_color": "yellow",
            "pipe_color": "green",
            "background": "day",
            "debug": False,
            "score_limit": -1,
        },
        "latency": {
            "train_values": [0],
            "train_mode": "episode_fixed",
            "train_sampling_mode": "uniform_all",
            "episodes_per_stage": 10,
            "randomize_within_stage": False,
            "within_episode_change_interval": 1,
            "within_episode_change_prob": 1.0,
        },
    }


def test_flappy_sf_gpu_adapter_reset_and_step_contract():
    pytest.importorskip("torch")
    env = FlappySampleFactoryGpuEnvAdapter(
        config=_flappy_sf_gpu_config(),
        noop_action=Action(value=0, name="noop", is_noop=True),
    )
    try:
        observation = env.reset(seed=123)

        assert observation.env_step == 0
        assert observation.data.shape == (12, 84, 84)
        assert observation.data.dtype == np.uint8
        assert observation.metadata["observation_mode"] == "sample_factory_gpu"
        assert "numeric_obs" in observation.metadata

        result = env.step(Action(value=0, name="noop", is_noop=True))

        assert result.observation is not None
        assert result.observation.env_step == 1
        assert result.observation.data.shape == (12, 84, 84)
        assert result.observation.data.dtype == np.uint8
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.truncated, bool)
        assert result.info["applied_action"] == 0
        assert result.info["applied_action_name"] == "noop"
        assert "numeric_obs" not in result.info
    finally:
        env.close()


def test_flappy_sf_gpu_adapter_leaves_latency_channel_to_latency_bench_builder():
    pytest.importorskip("torch")
    env = FlappySampleFactoryGpuEnvAdapter(
        config=_flappy_sf_gpu_config(obs_mode="latency_channel"),
        noop_action=Action(value=0, name="noop", is_noop=True),
    )
    try:
        observation = env.reset(seed=123)

        assert observation.data.shape == (12, 84, 84)
        assert "queue_latency" not in observation.metadata
        assert "frame_skip" not in observation.metadata
        assert "latency_frame_skip" not in observation.metadata
    finally:
        env.close()


def test_flappy_sf_gpu_adapter_reset_matches_sample_factory_gpu_observation():
    pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    adapter = FlappySampleFactoryGpuEnvAdapter(
        config=_flappy_sf_gpu_config(),
        noop_action=Action(value=0, name="noop", is_noop=True),
    )
    sf_env = make_flappy_sf_env(
        "latency_flappy",
        cfg=adapter.sf_cfg,
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
    )
    try:
        adapter_observation = adapter.reset(seed=123)
        sf_obs, _ = sf_env.reset(seed=123)
        sf_arr = sf_obs["obs"].detach().cpu().numpy()[0]

        assert np.array_equal(adapter_observation.data, sf_arr)
    finally:
        adapter.close()
        sf_env.close()
