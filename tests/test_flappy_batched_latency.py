from __future__ import annotations

import numpy as np


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
    infos = scheduler.finish_decision(np.asarray([True]), np.asarray([1], dtype=np.int64))
    assert infos[0]["latency_raw_frames"] == 1
    assert infos[0]["queued_action"] == 1
    assert infos[0]["applied_action"] == 0

    scheduler.begin_decision(np.asarray([0], dtype=np.int64), active_mask=np.asarray([True]))
    assert scheduler.raw_actions(np.asarray([True])).tolist() == [1]
    scheduler.record_raw_step(np.asarray([True]))
    infos = scheduler.finish_decision(np.asarray([True]), np.asarray([1], dtype=np.int64))
    assert infos[0]["queued_action"] == 0
    assert infos[0]["applied_action"] == 1


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

