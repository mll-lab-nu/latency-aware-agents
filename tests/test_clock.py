from __future__ import annotations

import pytest

from latency_bench.core.clock import EnvClock, FrameAccumulator


def test_env_clock_conversions_at_30_fps():
    clock = EnvClock(fps_env=30)

    assert clock.frame_ms == pytest.approx(33.3333333333)
    assert clock.step_to_time_ms(3) == pytest.approx(100.0)
    assert clock.time_ms_to_step_float(100.0) == pytest.approx(3.0)
    assert clock.latency_ms_to_frames_float(100.0) == pytest.approx(3.0)


def test_frame_accumulator_preserves_fractional_residuals():
    accumulator = FrameAccumulator()

    assert accumulator.consume(0.4) == 0
    assert accumulator.residual_frames == pytest.approx(0.4)
    assert accumulator.consume(0.4) == 0
    assert accumulator.residual_frames == pytest.approx(0.8)
    assert accumulator.consume(0.4) == 1
    assert accumulator.residual_frames == pytest.approx(0.2)

