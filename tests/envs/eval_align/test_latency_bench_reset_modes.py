from __future__ import annotations

import numpy as np
import pytest

from latency_bench.core.types import Action
from helpers import REPO_ROOT


FRAME_STACK = 4
SEED = 12345


# These tests lock the reset semantics before comparing policies:
# raw  = ALE reset + WarpFrame only.
# noop = SB3 NoopResetEnv and eval_core DemonAttackNoopResetWrapper.
# fire = SB3 NoopResetEnv followed by FireResetEnv, matching the current CULE reset contract.
def test_latency_bench_raw_reset_matches_plain_ale_reset_with_warp_frame():
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")

    import gymnasium as gym
    from stable_baselines3.common.atari_wrappers import WarpFrame

    _register_ale(gym)

    latency_env = _make_latency_env(reset_mode="raw")
    reference_env = WarpFrame(_make_plain_ale_env(gym), width=84, height=84)
    try:
        latency_env.reset(seed=SEED)
        reference_obs, _info = reference_env.reset(seed=SEED)

        _assert_latency_env_matches_reference(
            latency_env=latency_env,
            reference_env=reference_env,
            reference_obs=np.asarray(reference_obs[:, :, 0], dtype=np.uint8),
        )
    finally:
        latency_env.close()
        reference_env.close()


def test_latency_bench_noop_reset_matches_eval_core_and_sb3_noop_reset(monkeypatch):
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")

    import gymnasium as gym
    import torch
    from stable_baselines3.common.atari_wrappers import NoopResetEnv, WarpFrame
    from training.common.atari_cv2_obs import native_rgb_to_obs84

    monkeypatch.syspath_prepend(REPO_ROOT / "third_party" / "starVLA")
    eval_core = pytest.importorskip("starVLA.training.rl_games.eval_core", exc_type=ImportError)
    DemonAttackNoopResetWrapper = eval_core.DemonAttackNoopResetWrapper

    _register_ale(gym)

    latency_env = _make_latency_env(reset_mode="noop")
    sb3_noop_env = WarpFrame(NoopResetEnv(_make_plain_ale_env(gym), noop_max=30), width=84, height=84)
    eval_core_env = DemonAttackNoopResetWrapper(_make_plain_ale_env(gym), noop_max=30)
    try:
        latency_env.reset(seed=SEED)
        sb3_obs, _info = sb3_noop_env.reset(seed=SEED)
        eval_core_obs, _info = eval_core_env.reset(seed=SEED)
        eval_core_obs84 = native_rgb_to_obs84(torch.from_numpy(np.asarray(eval_core_obs, dtype=np.uint8))).cpu().numpy()

        _assert_latency_env_matches_reference(
            latency_env=latency_env,
            reference_env=sb3_noop_env,
            reference_obs=np.asarray(sb3_obs[:, :, 0], dtype=np.uint8),
        )
        assert np.array_equal(latency_env.gym_env.unwrapped.ale.getRAM(), eval_core_env.unwrapped.ale.getRAM())
        assert np.array_equal(latency_env.render_game_frame(), eval_core_env.unwrapped.ale.getScreenRGB())
        assert np.array_equal(_latest_latency_frame(latency_env.observe().data), eval_core_obs84)
    finally:
        latency_env.close()
        sb3_noop_env.close()
        eval_core_env.close()


def test_latency_bench_fire_reset_matches_sb3_noop_plus_fire_reset():
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")

    import gymnasium as gym
    from stable_baselines3.common.atari_wrappers import FireResetEnv, NoopResetEnv, WarpFrame

    _register_ale(gym)

    latency_env = _make_latency_env(reset_mode="fire")
    reference_env = WarpFrame(
        FireResetEnv(NoopResetEnv(_make_plain_ale_env(gym), noop_max=30)),
        width=84,
        height=84,
    )
    try:
        latency_env.reset(seed=SEED)
        reference_obs, _info = reference_env.reset(seed=SEED)

        _assert_latency_env_matches_reference(
            latency_env=latency_env,
            reference_env=reference_env,
            reference_obs=np.asarray(reference_obs[:, :, 0], dtype=np.uint8),
        )
    finally:
        latency_env.close()
        reference_env.close()


def test_latency_bench_reset_mode_frame_stack_advances_oldest_to_newest():
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")

    env = _make_latency_env(reset_mode="noop")
    try:
        env.reset(seed=SEED)
        reset_stack = np.asarray(env.observe().data, dtype=np.uint8).copy()
        reset_raw_stack = np.asarray(env.observe().metadata["env_raw_rgb_frame_stack"], dtype=np.uint8).copy()

        env.step(Action(value=0, name="NOOP", is_noop=True))
        step_stack = np.asarray(env.observe().data, dtype=np.uint8).copy()
        step_raw_stack = np.asarray(env.observe().metadata["env_raw_rgb_frame_stack"], dtype=np.uint8).copy()

        assert all(np.array_equal(reset_stack[0], reset_stack[index]) for index in range(FRAME_STACK))
        assert all(np.array_equal(reset_raw_stack[0], reset_raw_stack[index]) for index in range(FRAME_STACK))
        assert np.array_equal(step_stack[:-1], reset_stack[1:])
        assert np.array_equal(step_raw_stack[:-1], reset_raw_stack[1:])
    finally:
        env.close()


def _make_latency_env(*, reset_mode: str):
    from latency_bench.envs.demon_attack import DemonAttackEnvAdapter

    return DemonAttackEnvAdapter(
        env_fps=60.0,
        noop_action=Action(value=0, name="NOOP", is_noop=True),
        screen_size=84,
        noop_max=30,
        reset_mode=reset_mode,
        terminal_on_life_loss=False,
        clip_reward=False,
        frame_stack=FRAME_STACK,
        export_env_raw_rgb_frames=True,
    )


def _assert_latency_env_matches_reference(*, latency_env, reference_env, reference_obs: np.ndarray) -> None:
    assert np.array_equal(latency_env.gym_env.unwrapped.ale.getRAM(), reference_env.unwrapped.ale.getRAM())
    assert np.array_equal(latency_env.render_game_frame(), reference_env.unwrapped.ale.getScreenRGB())
    assert np.array_equal(_latest_latency_frame(latency_env.observe().data), reference_obs)
    assert np.array_equal(latency_env.observe().metadata["env_raw_rgb_frame_stack"][-1], reference_env.unwrapped.ale.getScreenRGB())


def _latest_latency_frame(observation_data) -> np.ndarray:
    return np.asarray(observation_data, dtype=np.uint8)[-1].copy()


def _make_plain_ale_env(gym):
    return gym.make(
        "ALE/DemonAttack-v5",
        frameskip=1,
        repeat_action_probability=0.0,
        obs_type="rgb",
    )


def _register_ale(gym) -> None:
    import ale_py

    gym.register_envs(ale_py)
