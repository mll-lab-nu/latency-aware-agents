from __future__ import annotations

import numpy as np


def test_latency_hot_path_returns_arrays_not_per_agent_infos():
    from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler

    scheduler = BatchedFlappyLatencyScheduler(
        batch_size=4,
        fixed_latency_raw_frames=1,
        env_fps=30,
        obs_fps=30,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        latency_distribution_spec=None,
        seeds=[1, 2, 3, 4],
        env_reset_seed_bases=[101, 102, 103, 104],
        add_latency_info=False,
    )
    scheduler.reset(seeds=[11, 12, 13, 14])
    active = np.ones(4, dtype=np.bool_)
    scheduler.begin_decision(np.asarray([0, 1, 0, 1], dtype=np.int64), active_mask=active)
    scheduler.raw_actions(active)
    scheduler.record_raw_step(active)

    result = scheduler.finish_decision(active, np.ones(4, dtype=np.int64))

    assert not isinstance(result, list)
    for field in (
        "queued_actions",
        "applied_actions",
        "latency_raw_frames",
        "ready_raw_frames",
        "raw_frames_advanced",
        "pending_counts",
    ):
        value = getattr(result, field)
        assert isinstance(value, np.ndarray)
        assert value.shape == (4,)


def test_core_step_hot_path_returns_scores_array_without_infos():
    from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore

    core = BatchedFlappyBirdCore(batch_size=4)
    core.reset_batch(seeds=[11, 12, 13, 14])

    result = core.step_batch(np.asarray([0, 1, 0, 1], dtype=np.int64))

    assert not hasattr(result, "infos")
    assert isinstance(result.scores, np.ndarray)
    assert result.scores.shape == (4,)


def test_metrics_no_done_hot_path_emits_no_dicts():
    from training.common.batched_metrics import BatchedEpisodeMetrics

    metrics = BatchedEpisodeMetrics(
        batch_size=4,
        frames_per_decision=1,
        metrics_path="",
        worker_index=0,
        vector_index=0,
        env_ids=[0, 1, 2, 3],
        issued_action_labels={0: "noop", 1: "flap"},
    )
    result = metrics.update(
        issued_actions=np.asarray([0, 1, 0, 1], dtype=np.int64),
        rewards=np.full(4, 0.1, dtype=np.float32),
        terminated=np.zeros(4, dtype=np.bool_),
        truncated=np.zeros(4, dtype=np.bool_),
        scores=np.zeros(4, dtype=np.int64),
        active_mask=np.ones(4, dtype=np.bool_),
        latency_raw_frames=np.zeros(4, dtype=np.int64),
        timeout_mask=np.zeros(4, dtype=np.bool_),
    )

    assert result == {}
