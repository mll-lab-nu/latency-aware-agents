from __future__ import annotations

import numpy as np
import pytest
from helpers import REPO_ROOT


SEED = 12345
ACTIONS = [0, 1, 2, 3, 4, 5, 0, 2]


def test_eval_core_frameskip_4_matches_four_frameskip_1_steps_from_same_ale_state():
    pytest.importorskip("ale_py")

    import gymnasium as gym

    _register_ale(gym)

    env_fs1 = _make_demon_attack_env(gym, frameskip=1)
    env_fs4 = _make_demon_attack_env(gym, frameskip=4)
    try:
        env_fs1.reset(seed=SEED)
        env_fs4.reset(seed=SEED + 1)

        # Reset wrappers advance a different number of raw frames when frameskip
        # changes. Clone/restore isolates the action-repeat contract itself.
        state = env_fs1.unwrapped.ale.cloneSystemState()
        env_fs4.unwrapped.ale.restoreSystemState(state)

        assert np.array_equal(env_fs1.unwrapped.ale.getScreenRGB(), env_fs4.unwrapped.ale.getScreenRGB())
        assert np.array_equal(env_fs1.unwrapped.ale.getRAM(), env_fs4.unwrapped.ale.getRAM())

        for action in ACTIONS:
            fs1_reward = 0.0
            fs1_terminated = False
            fs1_truncated = False
            for _raw_frame in range(4):
                fs1_obs, reward, terminated, truncated, _info = env_fs1.step(action)
                fs1_reward += float(reward)
                fs1_terminated = bool(fs1_terminated or terminated)
                fs1_truncated = bool(fs1_truncated or truncated)
                if bool(terminated or truncated):
                    break

            fs4_obs, fs4_reward, fs4_terminated, fs4_truncated, _info = env_fs4.step(action)

            assert fs1_reward == float(fs4_reward)
            assert fs1_terminated == bool(fs4_terminated)
            assert fs1_truncated == bool(fs4_truncated)
            assert np.array_equal(np.asarray(fs1_obs, dtype=np.uint8), np.asarray(fs4_obs, dtype=np.uint8))
            assert np.array_equal(env_fs1.unwrapped.ale.getScreenRGB(), env_fs4.unwrapped.ale.getScreenRGB())
            assert np.array_equal(env_fs1.unwrapped.ale.getRAM(), env_fs4.unwrapped.ale.getRAM())
    finally:
        env_fs1.close()
        env_fs4.close()


def test_eval_core_noop_reset_fs1_matches_sb3_noop_reset_without_fire_reset(monkeypatch):
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")

    import gymnasium as gym
    import torch
    from stable_baselines3.common.atari_wrappers import NoopResetEnv
    from training.common.atari_cv2_obs import native_rgb_to_obs84

    monkeypatch.syspath_prepend(REPO_ROOT / "third_party" / "starVLA")
    eval_core = pytest.importorskip("starVLA.training.rl_games.eval_core", exc_type=ImportError)
    DemonAttackNoopResetWrapper = eval_core.DemonAttackNoopResetWrapper

    _register_ale(gym)

    eval_core_env = DemonAttackNoopResetWrapper(_make_demon_attack_env(gym, frameskip=1), noop_max=30)
    sb3_noop_env = NoopResetEnv(_make_demon_attack_env(gym, frameskip=1), noop_max=30)
    try:
        eval_core_obs, _info = eval_core_env.reset(seed=SEED)
        sb3_noop_obs, _info = sb3_noop_env.reset(seed=SEED)

        assert np.array_equal(eval_core_env.unwrapped.ale.getRAM(), sb3_noop_env.unwrapped.ale.getRAM())
        assert np.array_equal(eval_core_env.unwrapped.ale.getScreenRGB(), sb3_noop_env.unwrapped.ale.getScreenRGB())
        assert np.array_equal(
            native_rgb_to_obs84(torch.from_numpy(np.asarray(eval_core_obs, dtype=np.uint8))).cpu().numpy(),
            native_rgb_to_obs84(torch.from_numpy(np.asarray(sb3_noop_obs, dtype=np.uint8))).cpu().numpy(),
        )
    finally:
        eval_core_env.close()
        sb3_noop_env.close()


def test_sb3_fire_reset_is_the_first_reset_difference_after_noop_reset():
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")

    import gymnasium as gym
    import torch
    from stable_baselines3.common.atari_wrappers import FireResetEnv, NoopResetEnv
    from training.common.atari_cv2_obs import native_rgb_to_obs84

    _register_ale(gym)

    noop_env = NoopResetEnv(_make_demon_attack_env(gym, frameskip=1), noop_max=30)
    fire_env = FireResetEnv(NoopResetEnv(_make_demon_attack_env(gym, frameskip=1), noop_max=30))
    try:
        noop_obs, _info = noop_env.reset(seed=SEED)
        fire_obs, _info = fire_env.reset(seed=SEED)

        noop_raw = noop_env.unwrapped.ale.getScreenRGB()
        fire_raw = fire_env.unwrapped.ale.getScreenRGB()
        noop_obs84 = native_rgb_to_obs84(torch.from_numpy(np.asarray(noop_obs, dtype=np.uint8))).cpu().numpy()
        fire_obs84 = native_rgb_to_obs84(torch.from_numpy(np.asarray(fire_obs, dtype=np.uint8))).cpu().numpy()

        assert _diff_count(noop_raw, fire_raw) == 153
        assert _diff_count(noop_env.unwrapped.ale.getRAM(), fire_env.unwrapped.ale.getRAM()) == 12
        assert _diff_count(noop_obs84, fire_obs84) == 26
    finally:
        noop_env.close()
        fire_env.close()


def _make_demon_attack_env(gym, *, frameskip: int):
    return gym.make(
        "ALE/DemonAttack-v5",
        frameskip=frameskip,
        repeat_action_probability=0.0,
        obs_type="rgb",
    )


def _diff_count(left, right) -> int:
    return int(np.count_nonzero(np.asarray(left) != np.asarray(right)))


def _register_ale(gym) -> None:
    import ale_py

    gym.register_envs(ale_py)
