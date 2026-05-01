from __future__ import annotations

from latency_bench.scheduler.decision import (
    EveryFrameDecisionScheduler,
    FixedFrameIntervalDecisionScheduler,
    FixedFrequencyDecisionScheduler,
)


def test_every_frame_decision_scheduler_observes_each_step():
    scheduler = EveryFrameDecisionScheduler()

    assert [scheduler.should_observe(step, 0.0) for step in range(5)] == [True] * 5


def test_fixed_frame_interval_decision_scheduler():
    scheduler = FixedFrameIntervalDecisionScheduler(interval_frames=2)

    assert [scheduler.should_observe(step, 0.0) for step in range(6)] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_fixed_frequency_decision_scheduler_maps_30_fps_10_hz_to_3_frames():
    scheduler = FixedFrequencyDecisionScheduler(fps_env=30, frequency_hz=10)

    assert scheduler.interval_frames == 3
    assert [scheduler.should_observe(step, 0.0) for step in range(7)] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]

