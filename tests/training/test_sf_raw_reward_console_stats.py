from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _clear_fake_sample_factory(monkeypatch):
    for name in list(sys.modules):
        if name == "sample_factory" or name.startswith("sample_factory."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _install_fake_sample_factory(monkeypatch, *, bad_non_batched_signature: bool = False):
    _clear_fake_sample_factory(monkeypatch)

    package_names = [
        "sample_factory",
        "sample_factory.algo",
        "sample_factory.algo.sampling",
        "sample_factory.algo.utils",
    ]
    for name in package_names:
        module = ModuleType(name)
        if name == "sample_factory":
            module.__version__ = "2.1.1"
        monkeypatch.setitem(sys.modules, name, module)

    misc = ModuleType("sample_factory.algo.utils.misc")
    misc.EPISODIC = "episodic"
    misc.POLICY_ID_KEY = "policy_id"
    monkeypatch.setitem(sys.modules, "sample_factory.algo.utils.misc", misc)

    non_batched_sampling = ModuleType("sample_factory.algo.sampling.non_batched_sampling")

    class ActorState:
        if bad_non_batched_signature:

            def record_env_step(self, reward, terminated, truncated, info):
                raise AssertionError("signature probe should reject this method")

        else:

            def record_env_step(self, reward, terminated: bool, truncated: bool, info, rollout_step):
                done = bool(terminated or truncated)
                self.curr_traj_buffer["rewards"][rollout_step] = float(reward)
                self.curr_traj_buffer["dones"][rollout_step] = done
                self.last_episode_reward += float(reward)
                self.last_episode_duration += self.env_info.frameskip if self.cfg.summaries_use_frameskip else 1
                if not done:
                    return None
                report = {
                    misc.POLICY_ID_KEY: self.curr_policy_id,
                    misc.EPISODIC: {
                        "reward": float(self.last_episode_reward),
                        "len": float(self.last_episode_duration),
                    },
                }
                if "episode_extra_stats" in info:
                    report[misc.EPISODIC]["episode_extra_stats"] = dict(info["episode_extra_stats"])
                self.last_episode_reward = 0.0
                self.last_episode_duration = 0.0
                return report

    non_batched_sampling.ActorState = ActorState
    monkeypatch.setitem(
        sys.modules,
        "sample_factory.algo.sampling.non_batched_sampling",
        non_batched_sampling,
    )

    batched_sampling = ModuleType("sample_factory.algo.sampling.batched_sampling")

    class BatchedVectorEnvRunner:
        def _process_env_step(self, rewards, dones_orig, infos):
            rewards_array = np.asarray(rewards, dtype=np.float64)
            dones_array = np.asarray(dones_orig, dtype=bool)
            self.curr_episode_reward += rewards_array
            self.curr_episode_len += 1
            finished_agents = [int(idx) for idx, done in enumerate(dones_array) if done]
            if not finished_agents:
                return []

            stats = {
                "reward": np.asarray([self.curr_episode_reward[idx] for idx in finished_agents]),
                "len": np.asarray([self.curr_episode_len[idx] for idx in finished_agents]),
            }
            for idx in finished_agents:
                self.curr_episode_reward[idx] = 0.0
                self.curr_episode_len[idx] = 0
            return [{misc.POLICY_ID_KEY: self.policy_id, misc.EPISODIC: stats}]

    batched_sampling.BatchedVectorEnvRunner = BatchedVectorEnvRunner
    monkeypatch.setitem(
        sys.modules,
        "sample_factory.algo.sampling.batched_sampling",
        batched_sampling,
    )
    return ActorState, BatchedVectorEnvRunner, misc


def _non_batched_state(ActorState, *, env_name: str = "latency_demon_attack"):
    state = ActorState.__new__(ActorState)
    state.curr_traj_buffer = {
        "rewards": np.zeros(1, dtype=np.float32),
        "dones": np.zeros(1, dtype=np.bool_),
    }
    state.cfg = SimpleNamespace(env=env_name, summaries_use_frameskip=False)
    state.env_info = SimpleNamespace(frameskip=1)
    state.curr_policy_id = 0
    state.last_episode_reward = 2.0
    state.last_episode_duration = 1.0
    return state


def _batched_runner(BatchedVectorEnvRunner, *, env_name: str = "latency_demon_attack"):
    runner = BatchedVectorEnvRunner.__new__(BatchedVectorEnvRunner)
    runner.cfg = SimpleNamespace(env=env_name, summaries_use_frameskip=False)
    runner.env_info = SimpleNamespace(frameskip=1)
    runner.vec_env = SimpleNamespace(num_agents=1)
    runner.policy_id = 4
    runner.curr_episode_reward = np.zeros(1, dtype=np.float64)
    runner.curr_episode_len = np.zeros(1, dtype=np.int32)
    runner.min_raw_rewards = np.ones(1, dtype=np.float64)
    runner.max_raw_rewards = np.ones(1, dtype=np.float64)
    return runner


def test_non_batched_report_main_reward_is_raw(monkeypatch):
    ActorState, _, misc = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_raw_reward_stats_compat import (
        ensure_sample_factory_raw_reward_console_stats_installed,
    )

    ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")
    state = _non_batched_state(ActorState)

    report = state.record_env_step(
        reward=1.0,
        terminated=True,
        truncated=False,
        info={
            "lives": 0,
            "episodic_return": 30.0,
            "episode_extra_stats": {
                "episode/raw_return": 30.0,
                "episode/return": 30.0,
            },
        },
        rollout_step=0,
    )

    stats = report[misc.EPISODIC]
    assert stats["reward"] == 30.0
    assert stats["episode_extra_stats"]["episode/raw_return"] == 30.0
    assert stats["episode_extra_stats"]["episode/return"] == 30.0
    assert "episode/clipped_return" not in stats["episode_extra_stats"]


def test_non_batched_life_loss_report_is_suppressed(monkeypatch):
    ActorState, _, _ = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_raw_reward_stats_compat import (
        ensure_sample_factory_raw_reward_console_stats_installed,
    )

    ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")
    state = _non_batched_state(ActorState)

    report = state.record_env_step(
        reward=1.0,
        terminated=True,
        truncated=False,
        info={"lives": 2, "is_true_episode_end": False},
        rollout_step=0,
    )

    assert report is None
    assert state.curr_traj_buffer["dones"][0]
    assert state.last_episode_reward == 2.0
    assert state.last_episode_duration == 2.0


def test_batched_truncation_overrides_false_episode_end_flag():
    from training.atari_episode_stats import is_true_batched_episode_end

    assert is_true_batched_episode_end(
        {
            "is_true_episode_end": False,
            "TimeLimit.truncated": True,
        }
    )


def test_batched_report_main_reward_is_raw_after_life_loss(monkeypatch):
    _, BatchedVectorEnvRunner, misc = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_raw_reward_stats_compat import (
        ensure_sample_factory_raw_reward_console_stats_installed,
    )

    ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")
    runner = _batched_runner(BatchedVectorEnvRunner)

    life_loss_reports = runner._process_env_step(
        rewards=np.asarray([1.0]),
        dones_orig=np.asarray([True]),
        infos=[{"lives": 2, "is_true_episode_end": False}],
    )

    assert life_loss_reports == []
    assert runner.curr_episode_reward.tolist() == [1.0]
    assert runner.curr_episode_len.tolist() == [1]

    game_over_reports = runner._process_env_step(
        rewards=np.asarray([2.0]),
        dones_orig=np.asarray([True]),
        infos=[
            {
                "lives": 0,
                "is_true_episode_end": True,
                "episode_extra_stats": {
                    "episode/raw_return": 50.0,
                    "episode/return": 50.0,
                },
            }
        ],
    )

    stats = game_over_reports[0][misc.EPISODIC]
    assert stats["reward"].tolist() == [50.0]
    assert stats["len"].tolist() == [2]
    assert stats["episode/raw_return"].tolist() == [50.0]
    assert stats["episode/return"].tolist() == [50.0]
    assert "episode/clipped_return" not in stats


def test_batched_life_loss_report_is_suppressed_even_when_episode_metrics_present(monkeypatch):
    _, BatchedVectorEnvRunner, _ = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_raw_reward_stats_compat import (
        ensure_sample_factory_raw_reward_console_stats_installed,
    )

    ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")
    runner = _batched_runner(BatchedVectorEnvRunner)

    reports = runner._process_env_step(
        rewards=np.asarray([1.0]),
        dones_orig=np.asarray([True]),
        infos=[
            {
                "lives": 2,
                "is_true_episode_end": False,
                "episodic_return": 10.0,
                "episode_extra_stats": {
                    "episode/raw_return": 10.0,
                    "episode/return": 10.0,
                },
            }
        ],
    )

    assert reports == []
    assert runner.curr_episode_reward.tolist() == [1.0]
    assert runner.curr_episode_len.tolist() == [1]


def test_patch_is_idempotent(monkeypatch):
    ActorState, BatchedVectorEnvRunner, _ = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_raw_reward_stats_compat import (
        ensure_sample_factory_raw_reward_console_stats_installed,
    )

    ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")
    first_actor_method = ActorState.record_env_step
    first_runner_method = BatchedVectorEnvRunner._process_env_step

    ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")

    assert ActorState.record_env_step is first_actor_method
    assert BatchedVectorEnvRunner._process_env_step is first_runner_method


def test_patch_does_not_affect_other_sample_factory_envs(monkeypatch):
    ActorState, _, misc = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_raw_reward_stats_compat import (
        ensure_sample_factory_raw_reward_console_stats_installed,
    )

    ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")
    state = _non_batched_state(ActorState, env_name="latency_flappy")

    report = state.record_env_step(
        reward=1.0,
        terminated=True,
        truncated=False,
        info={"lives": 0, "episodic_return": 30.0},
        rollout_step=0,
    )

    assert report[misc.EPISODIC]["reward"] == 3.0


def test_signature_change_fails_fast(monkeypatch):
    _install_fake_sample_factory(monkeypatch, bad_non_batched_signature=True)
    from training.common.sf_raw_reward_stats_compat import (
        ensure_sample_factory_raw_reward_console_stats_installed,
    )

    with pytest.raises(RuntimeError, match="record_env_step.*signature"):
        ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")


def test_unsupported_sample_factory_version_fails_fast(monkeypatch):
    _install_fake_sample_factory(monkeypatch)
    from training.common import sf_compat
    from training.common import sf_raw_reward_stats_compat as compat

    monkeypatch.setattr(sf_compat.metadata, "version", lambda package_name: "9.9.9")

    with pytest.raises(RuntimeError, match="Unsupported Sample Factory version"):
        compat.ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")


def test_demon_attack_env_factory_installs_worker_compat(monkeypatch):
    import gymnasium as gym

    from training.demon_attack_sf import integration

    class EmptyEnv(gym.Env):
        observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2), dtype=np.uint8)
        action_space = gym.spaces.Discrete(2)

        def reset(self, **kwargs):
            return np.zeros((2, 2), dtype=np.uint8), {}

        def step(self, action):
            raise AssertionError("not used")

    compat_calls: list[str] = []
    monkeypatch.setattr(
        integration,
        "ensure_sample_factory_raw_reward_console_stats_installed",
        lambda *, env_name: compat_calls.append(env_name),
    )
    monkeypatch.setattr(integration, "make_demon_attack_benchmark_env", lambda **kwargs: EmptyEnv())

    env = integration.make_demon_attack_sf_env(
        integration.SF_ENV_NAME,
        cfg=SimpleNamespace(
            seed=0,
            env_id="ALE/DemonAttack-v5",
            env_fps=60,
            obs_fps=15,
            simulator="cpu",
            screen_size=84,
            noop_max=30,
            reset_mode="fire",
            frame_stack=1,
            terminal_on_life_loss=True,
            clip_reward=True,
            repeat_action_probability=0.0,
            full_action_space=False,
            export_env_raw_rgb_frames=False,
            latency_type="zero",
            fixed_latency_ms=None,
            mean_latency_ms=None,
            std_latency_ms=None,
            min_latency_ms=None,
            max_latency_ms=None,
            latency_seed=None,
            simulated_worker_capacity=None,
            add_latency_info=False,
            hold_policy="hold",
            ordering_policy="latest_ready",
            max_pending_actions=None,
            max_episode_steps=0,
            episode_metrics_path="",
        ),
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
    )
    try:
        assert compat_calls == [integration.SF_ENV_NAME]
    finally:
        env.close()
