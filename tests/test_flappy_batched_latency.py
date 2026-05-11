from __future__ import annotations

import numpy as np
import pytest


class _LatencyDummyEnv:
    def __init__(self):
        import gymnasium as gym

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        del kwargs
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.asarray([int(action)], dtype=np.float32), 0.0, False, False, {}


def test_batched_latency_fixed_one_frame_then_noop_matches_scalar_action_timing():
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler

    scheduler = BatchedFlappyLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=1,
        env_fps=30,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="issue_order_fifo",
        latency_distribution_spec=None,
        seeds=[7],
        env_reset_seed_bases=[100],
        add_latency_info=True,
    )
    scheduler.reset(indices=[0], seeds=[123])

    scheduler.begin_decision(np.asarray([1], dtype=np.int64), active_mask=np.asarray([True]))
    assert scheduler.raw_actions(np.asarray([True])).tolist() == [0]
    scheduler.record_raw_step(np.asarray([True]))
    decision = scheduler.finish_decision(np.asarray([True]), np.asarray([1], dtype=np.int64))
    assert int(decision.latency_raw_frames[0]) == 1
    assert int(decision.queued_actions[0]) == 1
    assert int(decision.applied_actions[0]) == 0

    scheduler.begin_decision(np.asarray([0], dtype=np.int64), active_mask=np.asarray([True]))
    assert scheduler.raw_actions(np.asarray([True])).tolist() == [1]
    scheduler.record_raw_step(np.asarray([True]))
    decision = scheduler.finish_decision(np.asarray([True]), np.asarray([1], dtype=np.int64))
    assert int(decision.queued_actions[0]) == 0
    assert int(decision.applied_actions[0]) == 1


def test_batched_latency_reset_clears_pending_actions():
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler

    scheduler = BatchedFlappyLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=3,
        env_fps=30,
        obs_fps=30,
        hold_policy="hold",
        ordering_policy="issue_order_fifo",
        latency_distribution_spec=None,
        seeds=[7],
        env_reset_seed_bases=[100],
        add_latency_info=True,
    )
    scheduler.reset(indices=[0], seeds=[123])
    scheduler.begin_decision(np.asarray([1], dtype=np.int64), active_mask=np.asarray([True]))
    scheduler.reset(indices=[0], seeds=[456])

    scheduler.begin_decision(np.asarray([0], dtype=np.int64), active_mask=np.asarray([True]))
    assert scheduler.raw_actions(np.asarray([True])).tolist() == [0]


@pytest.mark.parametrize(
    "latency_distribution_spec",
    (
        {"type": "uniform", "min": 0, "max": 3},
        {"type": "normal", "mean": 1, "std": 1, "min": 0, "max": 3},
    ),
)
def test_batched_latency_sampled_distribution_matches_scalar_wrapper(latency_distribution_spec):
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler
    from training.sample_factory_common.latency_wrapper import ActionLatencyWrapper

    scheduler = BatchedFlappyLatencyScheduler(
        batch_size=1,
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=latency_distribution_spec,
        seeds=[17],
        env_reset_seed_bases=[101],
        add_latency_info=True,
    )
    scalar = ActionLatencyWrapper(
        _LatencyDummyEnv(),
        fixed_latency_raw_frames=0,
        env_fps=30,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=latency_distribution_spec,
        seed=17,
        env_reset_seed_base=101,
        add_latency_info=True,
    )
    scheduler.reset(indices=[0], seeds=[123])
    scalar.reset(seed=123)

    for action in [0, 1, 1, 0, 1, 0, 0, 1]:
        scheduler.begin_decision(np.asarray([action], dtype=np.int64), active_mask=np.asarray([True]))
        advanced = np.zeros(1, dtype=np.int64)
        for _ in range(scheduler.clock.obs_stride_raw_frames):
            scheduler.raw_actions(np.asarray([True]))
            scheduler.record_raw_step(np.asarray([True]))
            advanced[0] += 1
        decision = scheduler.finish_decision(np.asarray([True]), advanced)
        info = scheduler.decision_info_for_agent(0, decision)

        _, _, _, _, scalar_info = scalar.step(action)
        assert info["latency_raw_frames"] == scalar_info["latency_raw_frames"]
        assert info["queued_action"] == scalar_info["queued_action"]
        assert info["ready_raw_frame"] == scalar_info["ready_raw_frame"]
        assert info["applied_action"] == scalar_info["applied_action"]
        assert info["decision_step_idx"] == scalar_info["decision_step_idx"]
