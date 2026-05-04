from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import gymnasium as gym
import numpy as np
import torch


class _ScriptedAtariInfoEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)
        self.index = 0
        self.reset_count = 0

    def reset(self, *, seed=None, options=None):
        self.reset_count += 1
        return np.full((2, 2), self.reset_count, dtype=np.uint8), {}

    def step(self, action):
        if self.index >= len(self.steps):
            raise AssertionError("script exhausted")
        reward, terminated, truncated, info = self.steps[self.index]
        self.index += 1
        obs = np.full((2, 2), self.index + 10, dtype=np.uint8)
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)


class _SubprocScriptedSfEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 2, 2), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)
        self.index = 0
        self.reset_count = 0

    def reset(self, *, seed=None, options=None):
        self.reset_count += 1
        return np.full((1, 2, 2), self.reset_count, dtype=np.uint8), {
            "reset_seed": seed,
            "worker_env_pid": os.getpid(),
        }

    def step(self, action):
        if self.index >= len(self.steps):
            raise AssertionError("script exhausted")
        reward, terminated, truncated, info = self.steps[self.index]
        self.index += 1
        obs = np.full((1, 2, 2), self.index + 10, dtype=np.uint8)
        step_info = dict(info)
        step_info.setdefault("raw_reward", float(reward))
        step_info.setdefault("transformed_reward", float(np.sign(float(reward))))
        step_info["worker_env_pid"] = os.getpid()
        return obs, float(np.sign(float(reward))), bool(terminated), bool(truncated), step_info


class _FakeActorCritic:
    def eval(self):
        return self

    def model_to_device(self, device):
        self.device = torch.device(device)

    def normalize_obs(self, obs):
        return obs

    def device_for_input_tensor(self, _key):
        return torch.device("cpu")

    def type_for_input_tensor(self, _key):
        return torch.float32

    def __call__(self, obs, rnn_states):
        batch = int(next(iter(obs.values())).shape[0])
        return {
            "actions": torch.zeros((batch, 1), dtype=torch.int64),
            "new_rnn_states": torch.zeros_like(rnn_states),
        }


def _patch_queue_eval_sample_factory(monkeypatch):
    import sample_factory.cfg.arguments as sf_arguments
    import sample_factory.model.actor_critic as sf_actor_critic
    import sample_factory.model.model_utils as sf_model_utils

    from training.sample_factory_common import queue_eval

    monkeypatch.setattr(sf_arguments, "load_from_checkpoint", lambda cfg: cfg)
    monkeypatch.setattr(sf_actor_critic, "create_actor_critic", lambda *_args, **_kwargs: _FakeActorCritic())
    monkeypatch.setattr(sf_model_utils, "get_rnn_size", lambda _cfg: 1)
    monkeypatch.setattr(queue_eval, "load_sf_policy_state_dict", lambda *_args, **_kwargs: None)
    return queue_eval


def _demon_attack_queue_eval_cfg(integration, **overrides):
    values = {
        "env": integration.SF_ENV_NAME,
        "device": "cpu",
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
        "eval_episodes": 1,
        "eval_parallel_envs": 1,
        "eval_max_steps": 10,
        "eval_deterministic": False,
        "eval_raw_reward": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_demon_attack_eval_done_ignores_life_loss_pseudo_terminal():
    from training.sample_factory_common.queue_eval import _demon_attack_eval_episode_result

    result = _demon_attack_eval_episode_result(
        {
            "is_true_episode_end": False,
            "episodic_raw_return": 30.0,
            "episodic_clipped_return": 2.0,
            "episodic_return": 30.0,
            "episodic_length": 3,
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=True,
    )

    assert result is None


def test_demon_attack_eval_done_counts_life_loss_when_terminal_on_life_loss_enabled():
    from training.sample_factory_common.queue_eval import _demon_attack_eval_episode_result

    result = _demon_attack_eval_episode_result(
        {
            "is_true_episode_end": False,
            "episodic_raw_return": 30.0,
            "episodic_return": 30.0,
            "episodic_length": 3,
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=True,
        count_life_loss_terminal=True,
    )

    assert result == {
        "return": 30.0,
        "raw_return": 30.0,
        "clipped_return": 30.0,
        "length": 3,
    }


def test_demon_attack_eval_done_uses_episode_metrics_wrapper_keys_for_raw_scoring():
    from training.sample_factory_common.queue_eval import _demon_attack_eval_episode_result

    result = _demon_attack_eval_episode_result(
        {
            "is_true_episode_end": True,
            "episodic_raw_return": 30.0,
            "episodic_clipped_return": 2.0,
            "episodic_return": 30.0,
            "episodic_length": 3,
            "episode_extra_stats": {
                "episode/raw_return": 30.0,
                "episode/clipped_return": 2.0,
            },
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=True,
    )

    assert result == {
        "return": 30.0,
        "raw_return": 30.0,
        "clipped_return": 2.0,
        "length": 3,
    }


def test_demon_attack_eval_done_accepts_lives_zero_game_over_with_extra_stats():
    from training.sample_factory_common.queue_eval import _demon_attack_eval_episode_result

    result = _demon_attack_eval_episode_result(
        {
            "lives": 0,
            "episode_extra_stats": {
                "episode/raw_return": 50.0,
                "episode/clipped_return": 4.0,
                "episode/length": 8,
            },
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=False,
    )

    assert result == {
        "return": 4.0,
        "raw_return": 50.0,
        "clipped_return": 4.0,
        "length": 8,
    }


def test_demon_attack_eval_done_accepts_timelimit_truncated_timeout_with_extra_stats():
    from training.sample_factory_common.queue_eval import _demon_attack_eval_episode_result

    result = _demon_attack_eval_episode_result(
        {
            "TimeLimit.truncated": True,
            "episode_extra_stats": {
                "episode/raw_return": 12.0,
                "episode/clipped_return": 3.0,
                "episode/length": 4,
                "episode/truncated": 1.0,
                "episode/timeout": 1.0,
            },
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=True,
    )

    assert result == {
        "return": 12.0,
        "raw_return": 12.0,
        "clipped_return": 3.0,
        "length": 4,
        "timeout": True,
    }


def test_demon_attack_raw_eval_rejects_timeout_without_raw_stats():
    from training.sample_factory_common.queue_eval import _demon_attack_eval_episode_result

    with pytest.raises(ValueError, match="requires raw episode stats"):
        _demon_attack_eval_episode_result(
            {
                "TimeLimit.truncated": True,
                "truncated": True,
            },
            fallback_return=999.0,
            fallback_length=99,
            eval_raw_reward=True,
        )


def test_demon_attack_eval_budget_timeout_info_uses_raw_accumulator_not_fallback():
    from training.sample_factory_common.queue_eval import (
        _demon_attack_eval_budget_timeout_info,
        _demon_attack_eval_episode_result,
    )

    result = _demon_attack_eval_episode_result(
        _demon_attack_eval_budget_timeout_info(raw_return=12.0, clipped_return=12.0, length=4),
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=True,
    )

    assert result == {
        "return": 12.0,
        "raw_return": 12.0,
        "clipped_return": 12.0,
        "length": 4,
        "timeout": True,
    }


def test_demon_attack_timeout_count_uses_eval_budget_not_short_truncated_info():
    from training.sample_factory_common.queue_eval import (
        _demon_attack_eval_episode_result,
        _demon_attack_eval_step_budget_timeout,
    )

    short_truncated = _demon_attack_eval_episode_result(
        {
            "TimeLimit.truncated": True,
            "episode_extra_stats": {
                "episode/raw_return": 12.0,
                "episode/clipped_return": 3.0,
                "episode/length": 4,
                "episode/truncated": 1.0,
                "episode/timeout": 1.0,
            },
        },
        fallback_return=999.0,
        fallback_length=99,
        eval_raw_reward=True,
    )
    assert short_truncated is not None

    assert (
        _demon_attack_eval_step_budget_timeout(
            short_truncated,
            eval_max_steps=3600,
            timed_out_by_eval_budget=False,
        )
        is False
    )
    assert (
        _demon_attack_eval_step_budget_timeout(
            short_truncated,
            eval_max_steps=4,
            timed_out_by_eval_budget=True,
        )
        is True
    )


def test_demon_attack_vector_backend_defaults_to_serial_for_single_eval_env(monkeypatch):
    from training.sample_factory_common.queue_eval import _demon_attack_vector_backend

    monkeypatch.delenv("DEMON_EVAL_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("DEMON_EVAL_NUM_WORKERS", raising=False)

    assert _demon_attack_vector_backend(eval_parallel_envs=1) == "serial"


def test_demon_attack_vector_backend_defaults_to_subproc_for_parallel_eval_env(monkeypatch):
    from training.sample_factory_common.queue_eval import _demon_attack_vector_backend

    monkeypatch.delenv("DEMON_EVAL_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("DEMON_EVAL_NUM_WORKERS", raising=False)

    assert _demon_attack_vector_backend(eval_parallel_envs=4) == "subproc"


def test_demon_attack_vector_backend_debug_override_can_force_serial(monkeypatch):
    from training.sample_factory_common.queue_eval import _demon_attack_vector_backend

    monkeypatch.setenv("DEMON_EVAL_VECTOR_BACKEND", "serial")
    monkeypatch.delenv("DEMON_EVAL_NUM_WORKERS", raising=False)

    assert _demon_attack_vector_backend(eval_parallel_envs=4) == "serial"


def test_demon_attack_effective_workers_ignores_legacy_num_workers_env(monkeypatch):
    from training.sample_factory_common.queue_eval import _demon_attack_effective_parallel_envs

    monkeypatch.setenv("DEMON_EVAL_NUM_WORKERS", "1")

    assert _demon_attack_effective_parallel_envs(eval_parallel_envs=8, eval_episodes=3) == 3


def test_demon_attack_queue_eval_n1_uses_true_game_over_for_episode_accounting(monkeypatch, capsys):
    from training.demon_attack_sf import integration

    queue_eval = _patch_queue_eval_sample_factory(monkeypatch)
    monkeypatch.delenv("DEMON_EVAL_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("DEMON_EVAL_NUM_WORKERS", raising=False)
    monkeypatch.setattr(
        integration,
        "make_demon_attack_benchmark_env",
        lambda **_kwargs: _ScriptedAtariInfoEnv(
            [
                (1.0, True, False, {"lives": 2}),
                (2.0, True, False, {"lives": 0}),
            ]
        ),
    )

    cfg = _demon_attack_queue_eval_cfg(integration)

    result = queue_eval.evaluate_queue_latency(cfg, latency_value=0)
    stdout = capsys.readouterr().out

    assert result["completed_episodes"] == 1
    assert result["returns"] == [3.0]
    assert result["raw_returns"] == [3.0]
    assert result["clipped_returns"] == [2.0]
    assert result["_meta"]["episode_ids"] == [0]
    assert result["_meta"]["vector_backend"] == "serial"
    assert result["_meta"]["started_episodes"] == 1
    assert result["_meta"]["completed_episodes"] == 1
    assert "env=demon_attack eval_parallel_envs=1 vector_backend=serial effective_workers=1 mp_start_method=" in stdout


def test_demon_attack_queue_eval_n4_reassigns_slots_and_completes_exact_episode_ids(monkeypatch):
    from training.demon_attack_sf import integration

    queue_eval = _patch_queue_eval_sample_factory(monkeypatch)
    monkeypatch.setenv("DEMON_EVAL_VECTOR_BACKEND", "serial")
    monkeypatch.delenv("DEMON_EVAL_NUM_WORKERS", raising=False)
    scripted_envs = [
        _ScriptedAtariInfoEnv(
            [
                (10.0, True, False, {"lives": 0}),
                (11.0, True, False, {"lives": 0}),
            ]
        ),
        _ScriptedAtariInfoEnv(
            [
                (20.0, True, False, {"lives": 0}),
                (21.0, True, False, {"lives": 0}),
            ]
        ),
        _ScriptedAtariInfoEnv([(30.0, True, False, {"lives": 0})]),
        _ScriptedAtariInfoEnv([(40.0, True, False, {"lives": 0})]),
    ]
    created_envs = []

    def make_scripted_env(**_kwargs):
        env = scripted_envs.pop(0)
        created_envs.append(env)
        return env

    monkeypatch.setattr(integration, "make_demon_attack_benchmark_env", make_scripted_env)

    cfg = _demon_attack_queue_eval_cfg(
        integration,
        eval_episodes=6,
        eval_parallel_envs=4,
        eval_max_steps=10,
    )

    result = queue_eval.evaluate_queue_latency(cfg, latency_value=0)

    assert len(created_envs) == 4
    assert result["completed_episodes"] == 6
    assert result["_meta"]["started_episodes"] == 6
    assert result["_meta"]["completed_episodes"] == 6
    assert result["_meta"]["effective_parallel_envs"] == 4
    assert result["_meta"]["episode_ids"] == [0, 1, 2, 3, 4, 5]
    assert sorted(result["_meta"]["episode_ids"]) == list(range(6))
    assert len(set(result["_meta"]["episode_ids"])) == 6
    assert len(result["returns"]) == 6
    assert result["total_frames"] == 6
    assert result["_meta"]["vector_backend"] == "serial"


def test_demon_attack_queue_eval_parallel_envs_gt_episodes_only_starts_requested_episodes(monkeypatch):
    from training.demon_attack_sf import integration

    queue_eval = _patch_queue_eval_sample_factory(monkeypatch)
    monkeypatch.setenv("DEMON_EVAL_VECTOR_BACKEND", "serial")
    monkeypatch.delenv("DEMON_EVAL_NUM_WORKERS", raising=False)
    scripted_envs = [
        _ScriptedAtariInfoEnv([(1.0, True, False, {"lives": 0})]),
        _ScriptedAtariInfoEnv([(2.0, True, False, {"lives": 0})]),
        _ScriptedAtariInfoEnv([(3.0, True, False, {"lives": 0})]),
    ]
    created_envs = []

    def make_scripted_env(**_kwargs):
        env = scripted_envs.pop(0)
        created_envs.append(env)
        return env

    monkeypatch.setattr(integration, "make_demon_attack_benchmark_env", make_scripted_env)

    cfg = _demon_attack_queue_eval_cfg(
        integration,
        eval_episodes=3,
        eval_parallel_envs=8,
        eval_max_steps=10,
    )

    result = queue_eval.evaluate_queue_latency(cfg, latency_value=0)

    assert len(created_envs) == 3
    assert result["completed_episodes"] == 3
    assert result["_meta"]["started_episodes"] == 3
    assert result["_meta"]["completed_episodes"] == 3
    assert result["_meta"]["effective_parallel_envs"] == 3
    assert result["_meta"]["episode_ids"] == [0, 1, 2]
    assert len(set(result["_meta"]["episode_ids"])) == 3
    assert len(result["returns"]) == 3
    assert result["total_frames"] == 3
    assert result["_meta"]["vector_backend"] == "serial"


def test_demon_attack_queue_eval_max_steps_is_per_episode_slot(monkeypatch):
    from training.demon_attack_sf import integration

    queue_eval = _patch_queue_eval_sample_factory(monkeypatch)
    monkeypatch.setenv("DEMON_EVAL_VECTOR_BACKEND", "serial")
    monkeypatch.delenv("DEMON_EVAL_NUM_WORKERS", raising=False)
    scripted_envs = [
        _ScriptedAtariInfoEnv(
            [
                (2.0, False, False, {"lives": 3}),
                (2.0, False, False, {"lives": 3}),
            ]
        ),
        _ScriptedAtariInfoEnv(
            [
                (3.0, False, False, {"lives": 3}),
                (3.0, False, False, {"lives": 3}),
            ]
        ),
    ]
    monkeypatch.setattr(integration, "make_demon_attack_benchmark_env", lambda **_kwargs: scripted_envs.pop(0))

    cfg = _demon_attack_queue_eval_cfg(
        integration,
        eval_episodes=2,
        eval_parallel_envs=2,
        eval_max_steps=2,
    )

    result = queue_eval.evaluate_queue_latency(cfg, latency_value=0)

    assert result["completed_episodes"] == 2
    assert result["timeout_episodes"] == 2
    assert result["hit_eval_step_budget"] is True
    assert result["lengths"] == [2, 2]
    assert result["returns"] == [4.0, 6.0]
    assert result["raw_returns"] == [4.0, 6.0]
    assert result["clipped_returns"] == [2.0, 2.0]
    assert result["_meta"]["episode_ids"] == [0, 1]
    assert result["total_frames"] == 4


def test_demon_attack_queue_eval_timeout_after_life_loss_does_not_reset_inactive_slot(monkeypatch):
    from training.demon_attack_sf import integration

    queue_eval = _patch_queue_eval_sample_factory(monkeypatch)
    monkeypatch.delenv("DEMON_EVAL_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("DEMON_EVAL_NUM_WORKERS", raising=False)
    scripted_env = _ScriptedAtariInfoEnv([(5.0, True, False, {"lives": 2})])
    monkeypatch.setattr(integration, "make_demon_attack_benchmark_env", lambda **_kwargs: scripted_env)

    cfg = _demon_attack_queue_eval_cfg(
        integration,
        eval_episodes=1,
        eval_parallel_envs=1,
        eval_max_steps=1,
    )

    result = queue_eval.evaluate_queue_latency(cfg, latency_value=0)

    assert result["completed_episodes"] == 1
    assert result["timeout_episodes"] == 1
    assert result["_meta"]["episode_ids"] == [0]
    assert scripted_env.reset_count == 1


def test_demon_attack_queue_eval_subproc_backend_counts_true_game_over_and_timeout(monkeypatch, capsys):
    from training.demon_attack_sf import integration

    queue_eval = _patch_queue_eval_sample_factory(monkeypatch)

    def make_scripted_sf_env(_full_env_name, cfg=None, env_config=None, render_mode=None):
        del cfg, render_mode
        slot_idx = int(getattr(env_config, "vector_index", 0) or 0)
        if slot_idx == 0:
            return _SubprocScriptedSfEnv(
                [
                    (1.0, True, False, {"lives": 2, "is_true_episode_end": False}),
                    (
                        2.0,
                        True,
                        False,
                        {
                            "lives": 0,
                            "is_true_episode_end": True,
                            "episodic_raw_return": 3.0,
                            "episodic_clipped_return": 2.0,
                            "episodic_return": 3.0,
                            "episodic_length": 2,
                            "episode_extra_stats": {
                                "episode/raw_return": 3.0,
                                "episode/clipped_return": 2.0,
                                "episode/return": 3.0,
                                "episode/length": 2,
                            },
                        },
                    ),
                ]
            )
        return _SubprocScriptedSfEnv(
            [
                (3.0, False, False, {"lives": 3}),
                (3.0, False, False, {"lives": 3}),
            ]
        )

    monkeypatch.setattr(integration, "make_demon_attack_sf_env", make_scripted_sf_env)
    monkeypatch.delenv("DEMON_EVAL_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("DEMON_EVAL_NUM_WORKERS", raising=False)
    monkeypatch.setenv("DEMON_EVAL_MP_START_METHOD", "fork")

    cfg = _demon_attack_queue_eval_cfg(
        integration,
        eval_episodes=4,
        eval_parallel_envs=4,
        eval_max_steps=2,
    )

    result = queue_eval.evaluate_queue_latency(cfg, latency_value=0)
    stdout = capsys.readouterr().out

    assert result["completed_episodes"] == 4
    assert result["timeout_episodes"] == 3
    assert result["returns"] == [3.0, 6.0, 6.0, 6.0]
    assert result["raw_returns"] == [3.0, 6.0, 6.0, 6.0]
    assert result["clipped_returns"] == [2.0, 2.0, 2.0, 2.0]
    assert result["lengths"] == [2, 2, 2, 2]
    assert result["_meta"]["episode_ids"] == [0, 1, 2, 3]
    assert sorted(result["_meta"]["episode_ids"]) == [0, 1, 2, 3]
    assert len(set(result["_meta"]["episode_ids"])) == 4
    assert result["_meta"]["started_episodes"] == 4
    assert result["_meta"]["completed_episodes"] == 4
    assert result["_meta"]["effective_parallel_envs"] == 4
    assert result["_meta"]["vector_backend"] == "subproc"
    assert len(set(result["_meta"]["worker_pids"])) == 4
    assert all(pid != os.getpid() for pid in result["_meta"]["worker_pids"])
    assert "env=demon_attack eval_parallel_envs=4 vector_backend=subproc effective_workers=4 mp_start_method=fork" in stdout


def test_demon_attack_queue_eval_subproc_backend_parallel_envs_gt_episodes(monkeypatch):
    from training.demon_attack_sf import integration

    queue_eval = _patch_queue_eval_sample_factory(monkeypatch)

    def make_scripted_sf_env(_full_env_name, cfg=None, env_config=None, render_mode=None):
        del cfg, env_config, render_mode
        return _SubprocScriptedSfEnv(
            [
                (
                    4.0,
                    True,
                    False,
                    {
                        "lives": 0,
                        "is_true_episode_end": True,
                        "episodic_raw_return": 4.0,
                        "episodic_clipped_return": 1.0,
                        "episodic_return": 4.0,
                        "episodic_length": 1,
                        "episode_extra_stats": {
                            "episode/raw_return": 4.0,
                            "episode/clipped_return": 1.0,
                            "episode/return": 4.0,
                            "episode/length": 1,
                        },
                    },
                )
            ]
        )

    monkeypatch.setattr(integration, "make_demon_attack_sf_env", make_scripted_sf_env)
    monkeypatch.delenv("DEMON_EVAL_VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("DEMON_EVAL_NUM_WORKERS", raising=False)
    monkeypatch.setenv("DEMON_EVAL_MP_START_METHOD", "fork")

    cfg = _demon_attack_queue_eval_cfg(
        integration,
        eval_episodes=1,
        eval_parallel_envs=8,
        eval_max_steps=10,
    )

    result = queue_eval.evaluate_queue_latency(cfg, latency_value=0)

    assert result["completed_episodes"] == 1
    assert result["_meta"]["started_episodes"] == 1
    assert result["_meta"]["completed_episodes"] == 1
    assert result["_meta"]["effective_parallel_envs"] == 1
    assert result["_meta"]["episode_ids"] == [0]
    assert len(set(result["_meta"]["episode_ids"])) == 1
    assert result["_meta"]["vector_backend"] == "subproc"
    assert len(result["_meta"]["worker_pids"]) == 1
    assert result["_meta"]["worker_pids"][0] != os.getpid()
    assert result["returns"] == [4.0]
    assert result["raw_returns"] == [4.0]
    assert result["clipped_returns"] == [1.0]


def test_queue_eval_result_keeps_raw_and_clipped_fields_with_raw_compat_returns():
    from training.sample_factory_common.queue_eval import _build_queue_eval_result

    result = _build_queue_eval_result(
        latency_value=5,
        returns=[30.0, 50.0],
        lengths=[3, 8],
        eval_episodes=2,
        eval_max_steps=3600,
        total_frames=11,
        hit_eval_step_budget=False,
        raw_returns=[30.0, 50.0],
        clipped_returns=[2.0, 4.0],
        eval_raw_reward=True,
        env_name="latency_demon_attack",
    )

    assert result["returns"] == [30.0, 50.0]
    assert result["mean_return"] == 40.0
    assert result["raw_returns"] == [30.0, 50.0]
    assert result["clipped_returns"] == [2.0, 4.0]
    assert result["mean_raw_return"] == 40.0
    assert result["mean_clipped_return"] == 3.0
    assert result["std_raw_return"] == 10.0
    assert result["std_clipped_return"] == 1.0
    assert result["completed_episodes"] == 2
    assert "warning" not in result


def test_queue_eval_result_reports_timeout_episodes_without_missing_game_over_warning():
    from training.sample_factory_common.queue_eval import _build_queue_eval_result

    result = _build_queue_eval_result(
        latency_value=5,
        returns=[12.0],
        lengths=[3600],
        eval_episodes=1,
        eval_max_steps=3600,
        total_frames=3600,
        hit_eval_step_budget=True,
        raw_returns=[12.0],
        clipped_returns=[3.0],
        eval_raw_reward=True,
        env_name="latency_demon_attack",
        timeout_episodes=1,
    )

    assert result["returns"] == [12.0]
    assert result["raw_returns"] == [12.0]
    assert result["clipped_returns"] == [3.0]
    assert result["completed_episodes"] == 1
    assert result["requested_episodes"] == 1
    assert result["hit_eval_step_budget"] is True
    assert result["timeout_episodes"] == 1
    assert "warning" not in result
    assert result["mean_return"] == 12.0
    assert result["mean_raw_return"] == 12.0
    assert result["mean_clipped_return"] == 3.0


def test_demon_attack_eval_budget_timeout_info_carries_raw_metrics():
    from training.sample_factory_common.queue_eval import (
        _demon_attack_eval_budget_timeout_info,
        _demon_attack_eval_episode_result,
    )

    info = _demon_attack_eval_budget_timeout_info(
        accumulated_return=123.0,
        episode_length=3600,
        eval_raw_reward=True,
    )
    result = _demon_attack_eval_episode_result(
        info,
        fallback_return=-1.0,
        fallback_length=1,
        eval_raw_reward=True,
        count_life_loss_terminal=True,
    )

    assert result == {
        "return": 123.0,
        "raw_return": 123.0,
        "clipped_return": 123.0,
        "length": 3600,
        "timeout": True,
    }


def test_prepare_queue_eval_cfg_uses_eval_max_steps_as_eval_only_timelimit():
    from training.sample_factory_common.queue_eval import prepare_queue_eval_cfg

    cfg = prepare_queue_eval_cfg(
        SimpleNamespace(
            env="latency_flappy",
            eval_max_steps=123,
            max_episode_steps=0,
            eval_deterministic=True,
        ),
        latency_value=0,
    )

    assert cfg.max_episode_steps == 123
    assert cfg.cli_args["max_episode_steps"] == 123


def test_prepare_queue_eval_cfg_maps_flappy_gpu_eval_parallel_envs_to_batch_size():
    from training.sample_factory_common.queue_eval import prepare_queue_eval_cfg

    cfg = prepare_queue_eval_cfg(
        SimpleNamespace(
            env="latency_flappy",
            use_gpu_render=True,
            eval_episodes=100,
            eval_parallel_envs=4,
            eval_max_steps=123,
            max_episode_steps=0,
            eval_deterministic=True,
        ),
        latency_value=0,
    )

    assert cfg.gpu_render_batch_size == 4
    assert cfg.cli_args["gpu_render_batch_size"] == 4
    assert cfg.num_envs == 1
    assert cfg.num_envs_per_worker == 1


def test_prepare_queue_eval_cfg_caps_flappy_gpu_eval_parallel_envs_to_eval_episodes():
    from training.sample_factory_common.queue_eval import prepare_queue_eval_cfg

    cfg = prepare_queue_eval_cfg(
        SimpleNamespace(
            env="latency_flappy",
            use_gpu_render=True,
            eval_episodes=3,
            eval_parallel_envs=8,
            eval_max_steps=123,
            max_episode_steps=0,
            eval_deterministic=True,
        ),
        latency_value=0,
    )

    assert cfg.gpu_render_batch_size == 3
    assert cfg.cli_args["gpu_render_batch_size"] == 3


def test_flappy_queue_eval_episode_tracker_assigns_exact_unique_episode_ids():
    from training.sample_factory_common.queue_eval import _QueueEvalEpisodeTracker

    tracker = _QueueEvalEpisodeTracker(eval_episodes=10, eval_parallel_envs=4)

    assert tracker.started_episodes == 4
    assert tracker.completed_episodes == 0
    assert tracker.active_slots == 4
    assert tracker.slot_episode_id == [0, 1, 2, 3]

    completed = []
    while tracker.completed_episodes < tracker.eval_episodes:
        for slot in range(len(tracker.slot_episode_id)):
            if tracker.slot_episode_id[slot] is None:
                continue
            completed_episode_id, _next_episode_id = tracker.complete_slot(slot)
            completed.append(completed_episode_id)
            if tracker.completed_episodes == tracker.eval_episodes:
                break

    assert completed == list(range(10))
    assert tracker.started_episodes == 10
    assert tracker.completed_episodes == 10
    assert tracker.active_slots == 0
    assert sorted(completed) == list(range(10))
    assert len(set(completed)) == 10
    assert all(episode_id is None for episode_id in tracker.slot_episode_id)


def test_flappy_queue_eval_episode_tracker_single_slot_serial_order():
    from training.sample_factory_common.queue_eval import _QueueEvalEpisodeTracker

    tracker = _QueueEvalEpisodeTracker(eval_episodes=5, eval_parallel_envs=1)

    completed = []
    while tracker.completed_episodes < tracker.eval_episodes:
        completed_episode_id, next_episode_id = tracker.complete_slot(0)
        completed.append(completed_episode_id)
        if next_episode_id is not None:
            assert next_episode_id == tracker.started_episodes - 1

    assert completed == [0, 1, 2, 3, 4]
    assert tracker.started_episodes == 5
    assert tracker.completed_episodes == 5
    assert tracker.active_slots == 0


def test_flappy_queue_eval_episode_tracker_supports_more_parallel_envs_than_episodes():
    from training.sample_factory_common.queue_eval import _QueueEvalEpisodeTracker

    tracker = _QueueEvalEpisodeTracker(eval_episodes=3, eval_parallel_envs=8)

    assert tracker.started_episodes == 3
    assert tracker.active_slots == 3
    assert tracker.slot_episode_id == [0, 1, 2]

    completed = [tracker.complete_slot(slot)[0] for slot in range(3)]

    assert completed == [0, 1, 2]
    assert tracker.started_episodes == 3
    assert tracker.completed_episodes == 3
    assert tracker.active_slots == 0
    assert tracker.slot_episode_id == [None, None, None]


def test_prepare_queue_eval_cfg_does_not_convert_demon_attack_eval_budget_to_timelimit():
    from training.sample_factory_common.queue_eval import prepare_queue_eval_cfg

    cfg = prepare_queue_eval_cfg(
        SimpleNamespace(
            env="latency_demon_attack",
            eval_max_steps=123,
            max_episode_steps=0,
            eval_deterministic=True,
        ),
        latency_value=0,
    )

    assert cfg.max_episode_steps == 0
    assert "max_episode_steps" not in cfg.cli_args


def test_prepare_queue_eval_cfg_aligns_demon_attack_raw_eval_runtime_config():
    from training.sample_factory_common.queue_eval import prepare_queue_eval_cfg

    cfg = prepare_queue_eval_cfg(
        SimpleNamespace(
            env="latency_demon_attack",
            eval_raw_reward=True,
            terminal_on_life_loss=False,
            clip_reward=True,
            eval_max_steps=123,
            max_episode_steps=0,
            eval_deterministic=True,
        ),
        latency_value=0,
    )

    assert cfg.terminal_on_life_loss is True
<<<<<<< HEAD
    assert cfg.clip_reward is True
    assert cfg.cli_args["terminal_on_life_loss"] is True
    assert "clip_reward" not in cfg.cli_args


def test_write_and_log_queue_eval_result_preserve_raw_and_clipped_metrics(tmp_path):

    cfg = SimpleNamespace(output_dir=str(tmp_path), experiment="exp", seed=7, load_checkpoint_kind="best")
    result = {
        "latency": 5,
        "mean_return": 40.0,
        "std_return": 10.0,
        "mean_length": 5.5,
        "returns": [30.0, 50.0],
        "lengths": [3, 8],
        "raw_returns": [30.0, 50.0],
        "clipped_returns": [2.0, 4.0],
        "mean_raw_return": 40.0,
        "mean_clipped_return": 3.0,
        "std_raw_return": 10.0,
        "std_clipped_return": 1.0,
        "completed_episodes": 2,
        "timeout_episodes": 1,
        "requested_episodes": 2,
    }

    write_queue_eval_result(cfg, result)
    payload = json.loads((tmp_path / "queue_eval_latency_5.json").read_text(encoding="utf-8"))

    assert payload["raw_returns"] == [30.0, 50.0]
    assert payload["checkpoint_kind"] == "best"
    assert payload["clipped_returns"] == [2.0, 4.0]
    assert payload["mean_raw_return"] == 40.0
    assert payload["mean_clipped_return"] == 3.0
    assert payload["timeout_episodes"] == 1

    class _Run:
        def __init__(self):
            self.logged = None
            self.summary = {}

        def log(self, payload):
            self.logged = payload

    run = _Run()
    log_queue_eval_result(run, result)

    assert run.logged["queue_eval/mean_raw_return"] == 40.0
    assert run.logged["queue_eval/mean_clipped_return"] == 3.0
    assert run.logged["queue_eval/timeout_episodes"] == 1.0
    assert run.summary["queue_eval/latency_5/mean_raw_return"] == 40.0
    assert run.summary["queue_eval/latency_5/timeout_episodes"] == 1.0


def test_write_queue_eval_result_keeps_summary_schema_and_writes_optional_meta(tmp_path):
    from training.sample_factory_common.queue_eval import write_queue_eval_result

    cfg = SimpleNamespace(output_dir=str(tmp_path), experiment="exp", seed=7, load_checkpoint_kind="latest")
    result = {
        "latency": 2,
        "mean_return": 1.5,
        "std_return": 0.5,
        "mean_length": 3.5,
        "returns": [1.0, 2.0],
        "lengths": [3, 4],
        "completed_episodes": 2,
        "requested_episodes": 2,
        "_meta": {
            "eval_parallel_envs": 4,
            "started_episodes": 2,
            "completed_episodes": 2,
            "episode_ids": [0, 1],
            "slot_ids": [0, 1],
        },
    }

    write_queue_eval_result(cfg, result)

    payload = json.loads((tmp_path / "queue_eval_latency_2.json").read_text(encoding="utf-8"))
    assert "_meta" not in payload
    assert "episode_ids" not in payload
    assert payload["returns"] == [1.0, 2.0]
    assert payload["lengths"] == [3, 4]
    assert payload["completed_episodes"] == 2

    meta = json.loads((tmp_path / "queue_eval_latency_2.meta.json").read_text(encoding="utf-8"))
    assert meta["eval_parallel_envs"] == 4
    assert meta["episode_ids"] == [0, 1]
