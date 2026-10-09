"""Long-lived Gymnasium wrapper contracts: timing, action history and seeded delays."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from latency_bench.core.latency_distribution import parse_latency_distribution
from latency_bench.envs.latency_wrapper import ActionLatencyWrapper


class _ActionTraceEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        super().__init__()
        self.actions: list[int] = []
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.actions.clear()
        self.step_count = 0
        return np.array([0], dtype=np.uint8), {}

    def step(self, action):
        self.actions.append(int(action))
        self.step_count += 1
        obs = np.array([self.step_count], dtype=np.uint8)
        return obs, 1.0, False, False, {"raw_step": self.step_count}


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
def test_explicit_noop_and_early_episode_end(continuous, truncated):
    """Keep task-owned no-ops and stop at either Gymnasium episode boundary."""
    class Env(_ActionTraceEnv):
        def step(self, action):
            self.actions.append(np.array(action, copy=True))
            self.step_count += 1
            done = self.step_count == 3
            return (
                np.array([self.step_count], dtype=np.uint8),
                1.0,
                done and not truncated,
                done and truncated,
                {},
            )

    base = Env()
    if continuous:
        base.action_space = gym.spaces.Box(-5, 5, shape=(2,), dtype=np.float32)
        noop = np.array([2.0, -1.0], dtype=np.float32)
        action = np.array([1.0, 3.0], dtype=np.float32)
    else:
        base.action_space = gym.spaces.Discrete(3)
        noop, action = 2, 1
    expected_noop = np.array(noop, copy=True)
    env = ActionLatencyWrapper(
        base, noop_action=noop, fixed_latency_raw_frames=2,
        env_fps=60, obs_fps=10,
    )
    try:
        if continuous:
            noop[:] = 4  # Mutating the caller's array must not change the stored no-op.
        for _ in range(2):
            env.reset(seed=7)
            obs, reward, terminated, actual_truncated, info = env.step(action)
            np.testing.assert_array_equal(base.actions, [expected_noop, expected_noop, action])
            np.testing.assert_array_equal(obs, [3])
            assert reward == 3
            assert terminated == (not truncated)
            assert actual_truncated == truncated
            assert info["raw_frames_advanced"] == 3
    finally:
        env.close()


def test_public_wrapper_runs_without_reference_training_imports():
    """The public Gymnasium entrypoint must work without either reference framework."""
    subprocess.run(
        [sys.executable, "-c", """
import importlib.abc
import sys

class RejectReferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'training', 'sample_factory', 'starVLA'}:
            raise AssertionError(f'Public wrapper imported {fullname}')

sys.meta_path.insert(0, RejectReferenceImports())
import gymnasium as gym
from latency_bench.envs.latency_wrapper import ActionLatencyWrapper

env = ActionLatencyWrapper(
    gym.make('CartPole-v1'), noop_action=0, env_fps=50, obs_fps=25,
    fixed_latency_raw_frames=1,
)
try:
    env.reset(seed=7)
    assert env.step(1)[4]['raw_frames_advanced'] == 2
finally:
    env.close()
"""],
        check=True,
        capture_output=True,
        text=True,
    )



def test_deadly_tuple_action_history_uses_decision_boundary_action():

    class Env(gym.Env):
        action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(2)))
        observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 4, 4), dtype=np.uint8)

        def reset(self, *, seed=None, options=None):
            return np.zeros((3, 4, 4), dtype=np.uint8), {}

        def step(self, action):
            return np.zeros((3, 4, 4), dtype=np.uint8), 0.0, False, False, {}

    env = ActionLatencyWrapper(
        Env(), noop_action=(0, 0, 0, 0), fixed_latency_raw_frames=0, env_fps=35, obs_fps=8.75,
        hold_policy="hold", ordering_policy="latest_ready",
        simulated_worker_capacity=1, add_latency_info=False, action_history_decisions=8,
    )
    try:
        obs, _ = env.reset(seed=42)
        assert obs["transport"].shape == (120,)
        np.testing.assert_array_equal(obs["transport"], 0)
        obs, *_ = env.step((1, 0, 2, 1))
        np.testing.assert_array_equal(
            obs["transport"][-15:], [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        )
    finally:
        env.close()



def test_action_latency_wrapper_exposes_fixed_non_zero_latency():

    env = gym.make("CartPole-v1")
    wrapped = ActionLatencyWrapper(
        env,
        noop_action=0,
        fixed_latency_raw_frames=4,
        env_fps=60,
        obs_fps=60,
    )
    _, info = wrapped.reset(seed=123)
    assert info["latency_raw_frames"] == 4
    _, _, _, _, step_info = wrapped.step(0)
    assert step_info["latency_raw_frames"] == 4
    wrapped.close()



def test_action_latency_wrapper_uses_iid_profile_action_latency(tmp_path):


    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "profile.json").write_text(
        json.dumps(
            {
                "model_id": "model",
                "gpu_class": "gpu",
                "workload_id": "workload",
                "instance_id": "instance",
                "source_run_id": "run",
                "distribution_path": "latency_distribution.json",
            }
        ),
        encoding="utf-8",
    )
    (profile_dir / "latency_distribution.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "worker_slots": {
                    "0": {
                        "all": {
                            "spearman_rho": 0.0,
                            "observation_to_action_latency_ms": {
                                "quantile_levels": [0.0, 1.0],
                                "latency_ms": [20.0, 20.0],
                            },
                            "worker_service_time_ms": {
                                "quantile_levels": [0.0, 1.0],
                                "latency_ms": [10.0, 10.0],
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    env = gym.make("CartPole-v1")
    wrapped = ActionLatencyWrapper(
        env,
        noop_action=0,
        env_fps=30,
        obs_fps=30,
        profile_path=str(profile_dir),
        profile_worker_slot=0,
        seed=42,
    )
    try:
        wrapped.reset(seed=123)
        values = [wrapped.step(0)[4]["latency_raw_frames"] for _ in range(4)]
    finally:
        wrapped.close()

    assert values == [1, 1, 1, 1]



def test_temporal_action_latency_stream_continues_across_resets(monkeypatch):
    """Native profile training retains machine state between episodes."""
    from latency_bench.core.clock import EnvClock
    from latency_bench.latency.sample import LatencySample

    delays = iter([20.0, 40.0])
    sampler = SimpleNamespace(
        distributions={"0": {"all": {"observation_to_action_latency_ms": {"latency_ms": [20.0, 40.0]}, "worker_service_time_ms": {"latency_ms": [20.0, 40.0]}}}},
        sample=lambda context: LatencySample.from_latency_ms(
            next(delays), clock=EnvClock(env_fps=50, obs_fps=50)
        ),
    )
    monkeypatch.setattr("latency_bench.envs.latency_wrapper.build_profile_sampler", lambda *args, **kwargs: sampler)
    wrapped = ActionLatencyWrapper(
        _ActionTraceEnv(), noop_action=0, env_fps=50, obs_fps=50,
        profile_path="temporal-profile", profile_method="temporal", profile_worker_slot=0,
    )
    try:
        wrapped.reset(seed=123)
        first = wrapped.step(1)[4]["latency_raw_frames"]
        wrapped.reset(seed=123)
        second = wrapped.step(1)[4]["latency_raw_frames"]
    finally:
        wrapped.close()
    assert (first, second) == (1, 2)



def test_action_latency_wrapper_drops_busy_requests_and_reports_issued_applied():
    from latency_bench.core.clock import EnvClock
    from latency_bench.latency.sample import LatencySample

    class ServiceTimeSampler:
        def sample(self, context):
            return LatencySample.from_action_ready(
                action_ready_raw_frames=0,
                worker_service_raw_frames=3,
                worker_service_ms=100.0,
                clock=EnvClock(env_fps=30, obs_fps=30),
            )

    base_env = _ActionTraceEnv()
    wrapped = ActionLatencyWrapper(
        base_env,
        noop_action=0,
        env_fps=30,
        obs_fps=30,
        simulated_worker_capacity=1,
        add_latency_info=True,
    )
    wrapped.profile_sampler = ServiceTimeSampler()
    try:
        wrapped.reset(seed=123)
        _, _, _, _, first_info = wrapped.step(1)
        _, _, _, _, second_info = wrapped.step(0)
    finally:
        wrapped.close()

    assert base_env.actions == [1, 0]
    assert first_info["command_admitted"] is True
    assert first_info["issued_action"] == 1
    assert first_info["applied_action"] == 1
    assert first_info["worker_service_raw_frames"] == 3
    assert second_info["command_admitted"] is False
    assert second_info["observation_dropped"] is True
    assert second_info["issued_action"] == 0
    assert second_info["applied_action"] == 0



def test_action_latency_wrapper_replays_iid_profile_by_reset_seed(tmp_path):


    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "profile.json").write_text(
        json.dumps(
            {
                "model_id": "model",
                "gpu_class": "gpu",
                "workload_id": "workload",
                "instance_id": "instance",
                "source_run_id": "run",
                "distribution_path": "latency_distribution.json",
            }
        ),
        encoding="utf-8",
    )
    (profile_dir / "latency_distribution.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "worker_slots": {
                    "0": {
                        "all": {
                            "spearman_rho": 0.0,
                            "observation_to_action_latency_ms": {
                                "quantile_levels": [0.0, 1.0],
                                "latency_ms": [1.0, 99.0],
                            },
                            "worker_service_time_ms": {
                                "quantile_levels": [0.0, 1.0],
                                "latency_ms": [1.0, 99.0],
                            },
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    wrapped = ActionLatencyWrapper(
        gym.make("CartPole-v1"),
        noop_action=0,
        env_fps=30,
        obs_fps=30,
        profile_path=str(profile_dir),
        profile_worker_slot=0,
        seed=42,
    )

    try:
        wrapped.reset(seed=123)
        first = [wrapped.step(0)[4]["latency_raw_frames"] for _ in range(5)]
        wrapped.reset(seed=456)
        wrapped.step(0)
        wrapped.reset(seed=123)
        replay = [wrapped.step(0)[4]["latency_raw_frames"] for _ in range(5)]
    finally:
        wrapped.close()

    assert replay == first



def test_action_latency_wrapper_distribution_samples_deterministically():
    from gymnasium.spaces import Box, Discrete


    class NeverDoneEnv(gym.Env):
        observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space = Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

    spec = parse_latency_distribution({"type": "uniform", "min": 0, "max": 15})
    first = ActionLatencyWrapper(
        NeverDoneEnv(),
        noop_action=0,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        latency_distribution_spec=spec,
        seed=123,
    )
    second = ActionLatencyWrapper(
        NeverDoneEnv(),
        noop_action=0,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        latency_distribution_spec=spec,
        seed=123,
    )
    other = ActionLatencyWrapper(
        NeverDoneEnv(),
        noop_action=0,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        latency_distribution_spec=spec,
        seed=124,
    )

    try:
        first.reset()
        second.reset()
        other.reset()
        first_values = [first.step(0)[4]["latency_raw_frames"] for _ in range(20)]
        second_values = [second.step(0)[4]["latency_raw_frames"] for _ in range(20)]
        other_values = [other.step(0)[4]["latency_raw_frames"] for _ in range(20)]
    finally:
        first.close()
        second.close()
        other.close()

    assert first_values == second_values
    assert first_values != other_values
    assert all(0 <= value <= 15 for value in first_values)



def test_action_latency_wrapper_distribution_resets_episode_rng():
    from gymnasium.spaces import Box, Discrete


    class NeverDoneEnv(gym.Env):
        observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space = Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

    wrapped = ActionLatencyWrapper(
        NeverDoneEnv(),
        noop_action=0,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        latency_distribution_spec=parse_latency_distribution({"type": "uniform", "min": 0, "max": 15}),
        seed=99,
    )

    try:
        wrapped.reset()
        first_episode = [wrapped.step(0)[4]["latency_raw_frames"] for _ in range(20)]
        wrapped.reset()
        second_episode = [wrapped.step(0)[4]["latency_raw_frames"] for _ in range(20)]
    finally:
        wrapped.close()

    assert first_episode != second_episode



def test_action_latency_wrapper_mixed_distribution_samples_once_per_episode():
    from gymnasium.spaces import Box, Discrete


    class NeverDoneEnv(gym.Env):
        observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space = Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

    wrapped = ActionLatencyWrapper(
        NeverDoneEnv(),
        noop_action=0,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        latency_distribution_spec=parse_latency_distribution({"type": "uniform", "min": 0, "max": 15}),
        sample_distribution_per_episode=True,
        seed=99,
    )

    try:
        wrapped.reset()
        first_episode = [wrapped.step(0)[4]["latency_raw_frames"] for _ in range(20)]
        wrapped.reset()
        second_episode = [wrapped.step(0)[4]["latency_raw_frames"] for _ in range(20)]
    finally:
        wrapped.close()

    assert len(set(first_episode)) == 1
    assert len(set(second_episode)) == 1
    assert first_episode[0] != second_episode[0]



def test_action_latency_wrapper_distribution_rejects_too_small_pending_action_bound():


    env = gym.make("CartPole-v1")
    with pytest.raises(ValueError, match="max_pending_actions"):
        ActionLatencyWrapper(
            env,
            noop_action=0,
            fixed_latency_raw_frames=0,
            env_fps=30,
            obs_fps=30,
            latency_distribution_spec=parse_latency_distribution({"type": "uniform", "min": 0, "max": 5}),
            max_pending_actions=5,
        )
    env.close()



def test_action_latency_wrapper_advances_obs_stride_with_one_frame_noop():

    base_env = _ActionTraceEnv()
    wrapped = ActionLatencyWrapper(
        base_env,
        noop_action=0,
        fixed_latency_raw_frames=3,
        env_fps=60,
        obs_fps=10,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        add_latency_info=True,
    )

    try:
        wrapped.reset()
        obs, reward, terminated, truncated, info = wrapped.step(1)
    finally:
        wrapped.close()

    assert obs.tolist() == [6]
    assert reward == pytest.approx(6.0)
    assert not terminated
    assert not truncated
    assert base_env.actions == [0, 0, 0, 1, 0, 0]
    assert info["applied_source_raw_frame"] == -1
    assert info["applied_source_obs_id"] == -1
    assert info["latency_raw_frames"] == 3
    assert info["obs_stride_raw_frames"] == 6
    assert info["raw_frames_advanced"] == 6
    applied_events = wrapped.action_scheduler.get_applied_events()
    assert len(applied_events) == 1
    assert applied_events[0].obs_env_step == 0
    assert applied_events[0].ready_raw_frame == 3
    assert applied_events[0].apply_raw_frame == 3



def test_action_latency_wrapper_holds_action_across_raw_steps():

    base_env = _ActionTraceEnv()
    wrapped = ActionLatencyWrapper(
        base_env,
        noop_action=0,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=10,
        hold_policy="hold",
        ordering_policy="issue_order_fifo",
        add_latency_info=True,
    )

    try:
        wrapped.reset()
        _, reward, _, _, info = wrapped.step(1)
    finally:
        wrapped.close()

    assert reward == pytest.approx(3.0)
    assert base_env.actions == [1, 1, 1]
    assert info["obs_stride_raw_frames"] == 3



def test_action_latency_wrapper_auto_seeds_underlying_env_resets():
    from gymnasium.spaces import Box, Discrete


    class SeedEchoEnv(gym.Env):
        observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space = Discrete(2)

        def __init__(self):
            self.seeds: list[int | None] = []

        def reset(self, *, seed=None, options=None):
            self.seeds.append(seed)
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

    env = SeedEchoEnv()
    wrapped = ActionLatencyWrapper(
        env,
        noop_action=0,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        seed=11,
        env_reset_seed_base=500,
    )
    wrapped.reset()
    wrapped.reset()
    wrapped.reset(seed=999)
    assert env.seeds == [500, 501, 999]
