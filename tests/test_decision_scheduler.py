from __future__ import annotations

from latency_bench.core.clock import EnvClock
from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler


def test_observation_cadence_observes_each_step_when_stride_is_one():
    scheduler = ObservationCadenceDecisionScheduler(EnvClock(env_fps=30, obs_fps=30))

    assert [scheduler.should_observe(step, 0.0) for step in range(5)] == [True] * 5


def test_observation_cadence_observes_at_env_fps_over_obs_fps_stride():
    scheduler = ObservationCadenceDecisionScheduler(EnvClock(env_fps=60, obs_fps=10))

    assert [scheduler.should_observe(step, 0.0) for step in range(13)] == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
