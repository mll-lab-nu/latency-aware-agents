from __future__ import annotations

from latency_bench.core.types import Action
from latency_bench.envs.flappy import FlappyBirdEnvAdapter


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


def test_flappy_render_frame_returns_latest_rgb_frame():
    env = _make_adapter()
    try:
        observation = env.reset(seed=0)
        frame = env.render_frame()

        assert frame.shape == observation.data.shape
        assert frame.shape[2] == 3
    finally:
        env.close()

