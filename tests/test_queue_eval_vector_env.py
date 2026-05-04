from __future__ import annotations

import os
from types import SimpleNamespace

import gymnasium as gym
import numpy as np


class _ScriptedAtariInfoEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)
        self.index = 0
        self.reset_count = 0
        self.reset_seeds: list[int | None] = []

    def reset(self, *, seed=None, options=None):
        self.reset_count += 1
        self.reset_seeds.append(seed)
        return np.full((2, 2), self.reset_count, dtype=np.uint8), {}

    def step(self, action):
        if self.index >= len(self.steps):
            raise AssertionError("script exhausted")
        reward, terminated, truncated, info = self.steps[self.index]
        self.index += 1
        obs = np.full((2, 2), self.index + 10, dtype=np.uint8)
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)


class _PidReportingSfEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 2, 2), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, slot_idx: int):
        super().__init__()
        self.slot_idx = int(slot_idx)
        self.reset_count = 0
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.reset_count += 1
        return np.full((1, 2, 2), self.reset_count + self.slot_idx, dtype=np.uint8), {
            "slot_idx": self.slot_idx,
            "reset_seed": seed,
        }

    def step(self, action):
        self.step_count += 1
        obs = np.full((1, 2, 2), self.step_count + 10 + self.slot_idx, dtype=np.uint8)
        info = {
            "slot_idx": self.slot_idx,
            "worker_env_pid": os.getpid(),
            "raw_reward": float(self.slot_idx + 1),
            "transformed_reward": 1.0,
        }
        return obs, 1.0, False, False, info


def _demon_cfg(**overrides):
    values = {
        "env": "latency_demon_attack",
        "env_gpu_actions": False,
        "env_gpu_observations": False,
        "env_frameskip": 1,
        "seed": 0,
        "env_id": "ALE/DemonAttack-v5",
        "screen_size": 84,
        "noop_max": 0,
        "action_repeat": 1,
        "frame_stack": 1,
        "terminal_on_life_loss": True,
        "clip_reward": True,
        "repeat_action_probability": 0.0,
        "full_action_space": False,
        "queue_latency_values": "0",
        "latency_mode": "episode_fixed",
        "latency_sampling_mode": "uniform_all",
        "episodes_per_stage": 10,
        "randomize_within_stage": False,
        "within_episode_change_interval": 1,
        "within_episode_change_prob": 1.0,
        "add_latency_info": False,
        "max_episode_steps": 0,
        "episode_metrics_path": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_demon_attack_queue_eval_vector_env_n1_matches_sample_factory_single_env_auto_reset(monkeypatch):
    from training.demon_attack_sf import integration
    from training.sample_factory_common.queue_eval_vector_env import DemonAttackQueueEvalVectorEnv

    steps = [
        (0.5, False, False, {"lives": 3}),
        (2.0, True, False, {"lives": 0}),
    ]
    scripted_envs = [
        _ScriptedAtariInfoEnv(steps),
        _ScriptedAtariInfoEnv(steps),
    ]

    def make_scripted_env(**_kwargs):
        return scripted_envs.pop(0)

    monkeypatch.setattr(integration, "_ENVS_REGISTERED", False)
    monkeypatch.setattr(integration, "make_demon_attack_benchmark_env", make_scripted_env)

    cfg = _demon_cfg()
    old_env = integration.make_reference_env(cfg)
    new_env = DemonAttackQueueEvalVectorEnv(cfg=cfg, num_slots=1)
    try:
        old_obs, old_infos = old_env.reset(seed=123)
        new_obs, new_infos = new_env.reset(seed=123)

        assert old_obs["obs"].shape == new_obs["obs"].shape == (1, 1, 2, 2)
        assert old_infos == new_infos

        action = np.asarray([0], dtype=np.int64)
        old_obs, old_rew, old_term, old_trunc, old_infos = old_env.step(action)
        new_obs, new_rew, new_term, new_trunc, new_infos = new_env.step(action)

        assert old_obs["obs"].shape == new_obs["obs"].shape == (1, 1, 2, 2)
        assert old_rew.tolist() == new_rew.tolist() == [1.0]
        assert old_term.tolist() == new_term.tolist() == [False]
        assert old_trunc.tolist() == new_trunc.tolist() == [False]
        assert old_infos == new_infos

        old_obs, old_rew, old_term, old_trunc, old_infos = old_env.step(action)
        new_obs, new_rew, new_term, new_trunc, new_infos = new_env.step(action)

        assert old_obs["obs"].shape == new_obs["obs"].shape == (1, 1, 2, 2)
        assert old_rew.tolist() == new_rew.tolist() == [1.0]
        assert old_term.tolist() == new_term.tolist() == [True]
        assert old_trunc.tolist() == new_trunc.tolist() == [False]
        assert old_infos[0]["is_true_episode_end"] is True
        assert new_infos[0]["is_true_episode_end"] is True
        assert old_infos[0]["episodic_raw_return"] == new_infos[0]["episodic_raw_return"] == 2.5
        assert old_infos[0]["episodic_clipped_return"] == new_infos[0]["episodic_clipped_return"] == 2.0
        assert "reset_info" in old_infos[0]
        assert old_infos == new_infos
    finally:
        old_env.close()
        new_env.close()


def test_demon_attack_queue_eval_vector_env_manual_life_loss_reset_preserves_full_game_metrics(monkeypatch):
    from training.demon_attack_sf import integration
    from training.sample_factory_common.queue_eval_vector_env import DemonAttackQueueEvalVectorEnv

    scripted_env = _ScriptedAtariInfoEnv(
        [
            (1.0, True, False, {"lives": 2}),
            (2.0, True, False, {"lives": 0}),
        ]
    )

    monkeypatch.setattr(integration, "make_demon_attack_benchmark_env", lambda **_kwargs: scripted_env)

    env = DemonAttackQueueEvalVectorEnv(cfg=_demon_cfg(), num_slots=1, auto_reset_done=False)
    try:
        env.reset(seed=500)

        _obs, rew, terminated, truncated, infos = env.step(np.asarray([0], dtype=np.int64))

        assert rew.tolist() == [1.0]
        assert terminated.tolist() == [True]
        assert truncated.tolist() == [False]
        assert infos[0]["is_true_episode_end"] is False
        assert "episodic_raw_return" not in infos[0]
        assert "episodic_clipped_return" not in infos[0]
        assert "episode_extra_stats" not in infos[0]
        assert "reset_info" not in infos[0]

        env.reset_agents({0: None}, active_mask=[True])
        _obs, rew, terminated, truncated, infos = env.step(np.asarray([0], dtype=np.int64))

        assert rew.tolist() == [1.0]
        assert terminated.tolist() == [True]
        assert truncated.tolist() == [False]
        assert infos[0]["is_true_episode_end"] is True
        assert infos[0]["episodic_raw_return"] == 3.0
        assert infos[0]["episodic_clipped_return"] == 2.0
        assert infos[0]["episode_extra_stats"]["episode/raw_return"] == 3.0
        assert infos[0]["episode_extra_stats"]["episode/clipped_return"] == 2.0
        assert infos[0]["episode_extra_stats"]["episode/length"] == 2
        assert scripted_env.reset_seeds == [500, 104730]
    finally:
        env.close()


def test_demon_attack_queue_eval_subproc_vector_env_steps_in_worker_processes(monkeypatch):
    from training.demon_attack_sf import integration
    from training.sample_factory_common.queue_eval_vector_env import DemonAttackQueueEvalSubprocVectorEnv

    def make_scripted_sf_env(_full_env_name, cfg=None, env_config=None, render_mode=None):
        del cfg, render_mode
        return _PidReportingSfEnv(slot_idx=int(getattr(env_config, "vector_index", 0) or 0))

    monkeypatch.setattr(integration, "make_demon_attack_sf_env", make_scripted_sf_env)
    monkeypatch.setenv("DEMON_EVAL_MP_START_METHOD", "fork")

    env = DemonAttackQueueEvalSubprocVectorEnv(cfg=_demon_cfg(), num_slots=2)
    processes = list(env._processes)
    try:
        obs, infos = env.reset(seed=700)
        assert obs["obs"].shape == (2, 1, 2, 2)
        assert [info["reset_seed"] for info in infos] == [700, 701]

        _obs, rew, terminated, truncated, infos = env.step(np.asarray([0, 0], dtype=np.int64))

        assert rew.tolist() == [1.0, 1.0]
        assert terminated.tolist() == [False, False]
        assert truncated.tolist() == [False, False]
        worker_pids = [int(info["_queue_eval_worker_pid"]) for info in infos]
        assert len(set(worker_pids)) == 2
        assert all(pid != os.getpid() for pid in worker_pids)
        assert [info["worker_env_pid"] for info in infos] == worker_pids
    finally:
        env.close()

    assert all(not process.is_alive() for process in processes)
