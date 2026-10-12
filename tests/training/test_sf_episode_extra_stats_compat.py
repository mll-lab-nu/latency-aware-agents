from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _clear_fake_sample_factory(monkeypatch):
    for name in list(sys.modules):
        if name == "sample_factory" or name.startswith("sample_factory."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def _install_fake_sample_factory(monkeypatch, *, empty_reports: bool = False):
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
            module.__version__ = "2.1.3"
        monkeypatch.setitem(sys.modules, name, module)

    misc = ModuleType("sample_factory.algo.utils.misc")
    misc.EPISODIC = "episodic"
    misc.POLICY_ID_KEY = "policy_id"
    monkeypatch.setitem(sys.modules, "sample_factory.algo.utils.misc", misc)

    non_batched_sampling = ModuleType("sample_factory.algo.sampling.non_batched_sampling")

    class ActorState:
        def record_env_step(self, reward, terminated: bool, truncated: bool, info, rollout_step):
            raise AssertionError("generic batched tests should not call non-batched sampling")

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
                "existing": np.asarray([123.0 for _ in finished_agents]),
            }
            for idx in finished_agents:
                self.curr_episode_reward[idx] = 0.0
                self.curr_episode_len[idx] = 0
            if empty_reports:
                return []
            return [{misc.POLICY_ID_KEY: self.policy_id, misc.EPISODIC: stats}]

    batched_sampling.BatchedVectorEnvRunner = BatchedVectorEnvRunner
    monkeypatch.setitem(
        sys.modules,
        "sample_factory.algo.sampling.batched_sampling",
        batched_sampling,
    )
    return ActorState, BatchedVectorEnvRunner, misc


def _runner(BatchedVectorEnvRunner, *, env_name: str = "latency_flappy", num_agents: int = 1):
    runner = BatchedVectorEnvRunner.__new__(BatchedVectorEnvRunner)
    runner.cfg = SimpleNamespace(env=env_name, summaries_use_frameskip=False)
    runner.env_info = SimpleNamespace(frameskip=1)
    runner.vec_env = SimpleNamespace(num_agents=num_agents)
    runner.policy_id = 7
    runner.curr_episode_reward = np.zeros(num_agents, dtype=np.float64)
    runner.curr_episode_len = np.zeros(num_agents, dtype=np.int32)
    runner.min_raw_rewards = np.full(num_agents, -1.0, dtype=np.float64)
    runner.max_raw_rewards = np.full(num_agents, 1.0, dtype=np.float64)
    return runner


def _episode_info(**extra):
    return {
        "episode_extra_stats": {
            "episode/return": 30.0,
            "episode/raw_return": np.float32(30.0),
            "episode/length": 3,
            "episode/non_numeric": "ignored",
            "existing": 999.0,
            **extra,
        }
    }


def test_forwards_numeric_episode_extra_stats_from_batched_list_infos(monkeypatch):
    _, BatchedVectorEnvRunner, misc = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_episode_extra_stats_compat import (
        ensure_sample_factory_episode_extra_stats_installed,
    )

    ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")
    runner = _runner(BatchedVectorEnvRunner)

    reports = runner._process_env_step(
        rewards=np.asarray([1.0]),
        dones_orig=np.asarray([True]),
        infos=[_episode_info()],
    )

    stats = reports[0][misc.EPISODIC]
    assert stats["episode/return"].tolist() == [30.0]
    assert stats["episode/raw_return"].tolist() == [30.0]
    assert stats["episode/length"].tolist() == [3.0]
    assert "episode/non_numeric" not in stats


def test_does_not_modify_original_reward_len_or_existing_metric_keys(monkeypatch):
    _, BatchedVectorEnvRunner, misc = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_episode_extra_stats_compat import (
        ensure_sample_factory_episode_extra_stats_installed,
    )

    ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")
    runner = _runner(BatchedVectorEnvRunner)

    reports = runner._process_env_step(
        rewards=np.asarray([1.0]),
        dones_orig=np.asarray([True]),
        infos=[_episode_info()],
    )

    stats = reports[0][misc.EPISODIC]
    assert stats["reward"].tolist() == [1.0]
    assert stats["len"].tolist() == [1]
    assert stats["existing"].tolist() == [123.0]


def test_disabled_env_name_does_not_forward_stats(monkeypatch):
    _, BatchedVectorEnvRunner, misc = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_episode_extra_stats_compat import (
        ensure_sample_factory_episode_extra_stats_installed,
    )

    ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")
    runner = _runner(BatchedVectorEnvRunner, env_name="latency_demon_attack")

    reports = runner._process_env_step(
        rewards=np.asarray([1.0]),
        dones_orig=np.asarray([True]),
        infos=[_episode_info()],
    )

    stats = reports[0][misc.EPISODIC]
    assert "episode/return" not in stats


def test_synthesizes_report_when_original_runner_returns_empty_reports(monkeypatch):
    _, BatchedVectorEnvRunner, misc = _install_fake_sample_factory(monkeypatch, empty_reports=True)
    from training.common.sf_episode_extra_stats_compat import (
        ensure_sample_factory_episode_extra_stats_installed,
    )

    ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")
    runner = _runner(BatchedVectorEnvRunner)

    reports = runner._process_env_step(
        rewards=np.asarray([2.0]),
        dones_orig=np.asarray([True]),
        infos=[_episode_info()],
    )

    assert len(reports) == 1
    assert reports[0][misc.POLICY_ID_KEY] == 7
    stats = reports[0][misc.EPISODIC]
    assert stats["reward"].tolist() == [2.0]
    assert stats["len"].tolist() == [1]
    assert stats["min_raw_reward"].tolist() == [-1.0]
    assert stats["max_raw_reward"].tolist() == [1.0]
    assert stats["episode/return"].tolist() == [30.0]


def test_repeated_install_is_idempotent_and_does_not_duplicate_stats(monkeypatch):
    _, BatchedVectorEnvRunner, misc = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_episode_extra_stats_compat import (
        ensure_sample_factory_episode_extra_stats_installed,
    )

    ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")
    first_method = BatchedVectorEnvRunner._process_env_step
    ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")
    second_method = BatchedVectorEnvRunner._process_env_step
    runner = _runner(BatchedVectorEnvRunner)

    reports = runner._process_env_step(
        rewards=np.asarray([1.0]),
        dones_orig=np.asarray([True]),
        infos=[_episode_info()],
    )

    assert first_method is second_method
    stats = reports[0][misc.EPISODIC]
    assert stats["episode/return"].tolist() == [30.0]


def test_repeated_install_after_other_wrapper_reuses_existing_generic_patch(monkeypatch):
    _, BatchedVectorEnvRunner, misc = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_episode_extra_stats_compat import (
        ensure_sample_factory_episode_extra_stats_installed,
    )
    from training.common.sf_raw_reward_stats_compat import (
        ensure_sample_factory_raw_reward_console_stats_installed,
    )

    ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")
    generic_method = BatchedVectorEnvRunner._process_env_step
    ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")
    raw_method = BatchedVectorEnvRunner._process_env_step
    ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")

    assert BatchedVectorEnvRunner._process_env_step is raw_method
    assert getattr(raw_method, "_latency_bench_original_process_env_step") is generic_method

    runner = _runner(BatchedVectorEnvRunner)
    reports = runner._process_env_step(
        rewards=np.asarray([1.0]),
        dones_orig=np.asarray([True]),
        infos=[_episode_info()],
    )

    stats = reports[0][misc.EPISODIC]
    assert stats["episode/return"].tolist() == [30.0]


def test_multiple_env_name_registrations_are_preserved(monkeypatch):
    _, BatchedVectorEnvRunner, misc = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_episode_extra_stats_compat import (
        ensure_sample_factory_episode_extra_stats_installed,
    )

    ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")
    ensure_sample_factory_episode_extra_stats_installed(env_name="latency_other")
    runner = _runner(BatchedVectorEnvRunner, env_name="latency_other")

    reports = runner._process_env_step(
        rewards=np.asarray([1.0]),
        dones_orig=np.asarray([True]),
        infos=[_episode_info()],
    )

    stats = reports[0][misc.EPISODIC]
    assert stats["episode/return"].tolist() == [30.0]


@pytest.mark.parametrize("install_order", ["generic_first", "raw_first"])
def test_generic_patch_and_demon_attack_raw_patch_can_coexist(monkeypatch, install_order):
    _, BatchedVectorEnvRunner, misc = _install_fake_sample_factory(monkeypatch)
    from training.common.sf_episode_extra_stats_compat import (
        ensure_sample_factory_episode_extra_stats_installed,
    )
    from training.common.sf_raw_reward_stats_compat import (
        ensure_sample_factory_raw_reward_console_stats_installed,
    )

    if install_order == "generic_first":
        ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")
        ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")
    else:
        ensure_sample_factory_raw_reward_console_stats_installed(env_name="latency_demon_attack")
        ensure_sample_factory_episode_extra_stats_installed(env_name="latency_flappy")
    runner = _runner(BatchedVectorEnvRunner, env_name="latency_demon_attack")

    life_loss_reports = runner._process_env_step(
        rewards=np.asarray([1.0]),
        dones_orig=np.asarray([True]),
        infos=[{"lives": 2, "is_true_episode_end": False}],
    )
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

    assert life_loss_reports == []
    stats = game_over_reports[0][misc.EPISODIC]
    assert stats["reward"].tolist() == [50.0]
    assert stats["len"].tolist() == [2]
    assert stats["episode/raw_return"].tolist() == [50.0]
    assert stats["episode/return"].tolist() == [50.0]
    assert "episode/clipped_return" not in stats
