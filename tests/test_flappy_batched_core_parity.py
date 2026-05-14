from __future__ import annotations

import numpy as np

from latency_bench.envs.flappy import ensure_flappy_gymnasium_on_path

ensure_flappy_gymnasium_on_path()

import gymnasium as gym  # noqa: E402

from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState  # noqa: E402


def _make_scalar_env(**overrides):
    kwargs = {
        "use_lidar": False,
        "normalize_obs": True,
        "audio_on": False,
        "pipe_gap": 100,
        "bird_color": "yellow",
        "pipe_color": "green",
        "background": "day",
        "debug": False,
        "render_mode": None,
    }
    kwargs.update(overrides)
    return gym.make("FlappyBird-v0", **kwargs)


def _assert_render_batch_equal(actual, expected):
    np.testing.assert_array_equal(actual.player_xy, expected.player_xy)
    np.testing.assert_array_equal(actual.player_indices, expected.player_indices)
    np.testing.assert_array_equal(actual.visible_rotations, expected.visible_rotations)
    np.testing.assert_array_equal(actual.upper_pipe_xy, expected.upper_pipe_xy)
    np.testing.assert_array_equal(actual.lower_pipe_xy, expected.lower_pipe_xy)
    np.testing.assert_array_equal(actual.ground_xy, expected.ground_xy)
    assert actual.screen_width == expected.screen_width
    assert actual.screen_height == expected.screen_height
    assert actual.bird_color == expected.bird_color
    assert actual.pipe_color == expected.pipe_color
    assert actual.background == expected.background


def test_batched_flappy_core_matches_scalar_env_for_mixed_actions_and_render_state():
    from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore

    seed = 123
    actions = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    scalar = _make_scalar_env()
    core = BatchedFlappyBirdCore(batch_size=1)
    try:
        scalar_obs, scalar_info = scalar.reset(seed=seed)
        core_obs, core_infos = core.reset_batch(seeds=[seed])

        np.testing.assert_allclose(core_obs[0], scalar_obs, rtol=0, atol=0)
        assert core_infos[0]["score"] == scalar_info["score"]
        _assert_render_batch_equal(
            core.render_state_batch(),
            BatchedFlappyRenderState.from_states([scalar.unwrapped.get_render_state()]),
        )

        for action in actions:
            scalar_obs, scalar_reward, scalar_term, scalar_trunc, scalar_info = scalar.step(action)
            result = core.step_batch(np.asarray([action], dtype=np.int64))

            np.testing.assert_allclose(result.observations[0], scalar_obs, rtol=0, atol=0)
            assert float(result.rewards[0]) == float(scalar_reward)
            assert bool(result.terminated[0]) is bool(scalar_term)
            assert bool(result.truncated[0]) is bool(scalar_trunc)
            assert int(result.scores[0]) == scalar_info["score"]
            _assert_render_batch_equal(
                core.render_state_batch(),
                BatchedFlappyRenderState.from_states([scalar.unwrapped.get_render_state()]),
            )
    finally:
        scalar.close()


def test_batched_flappy_core_keeps_agents_independent_inside_larger_batch():
    from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore

    seeds = [101, 202, 303, 404]
    action_rows = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.int64,
    )
    scalars = [_make_scalar_env() for _ in seeds]
    core = BatchedFlappyBirdCore(batch_size=len(seeds))
    try:
        scalar_obs = []
        scalar_infos = []
        for env, seed in zip(scalars, seeds):
            obs, info = env.reset(seed=seed)
            scalar_obs.append(obs)
            scalar_infos.append(info)
        core_obs, core_infos = core.reset_batch(seeds=seeds)
        np.testing.assert_allclose(core_obs, np.asarray(scalar_obs), rtol=0, atol=0)
        assert [info["score"] for info in core_infos] == [info["score"] for info in scalar_infos]

        for actions in action_rows:
            expected = [env.step(int(action)) for env, action in zip(scalars, actions)]
            result = core.step_batch(actions)
            np.testing.assert_allclose(
                result.observations,
                np.asarray([item[0] for item in expected]),
                rtol=0,
                atol=0,
            )
            np.testing.assert_array_equal(result.rewards, np.asarray([item[1] for item in expected], dtype=np.float64))
            np.testing.assert_array_equal(result.terminated, np.asarray([item[2] for item in expected], dtype=np.bool_))
            np.testing.assert_array_equal(result.truncated, np.asarray([item[3] for item in expected], dtype=np.bool_))
            assert [int(score) for score in result.scores] == [item[4]["score"] for item in expected]
    finally:
        for env in scalars:
            env.close()
