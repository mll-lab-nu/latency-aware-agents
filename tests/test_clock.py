from __future__ import annotations

import pytest

from latency_bench.core.clock import (
    EnvClock,
    FrameAccumulator,
    compute_obs_stride_raw_frames,
    ms_to_raw_frames,
    validate_clock_rates,
)


def test_env_clock_conversions_at_30_fps():
    clock = EnvClock(fps_env=30)

    assert clock.frame_ms == pytest.approx(33.3333333333)
    assert clock.step_to_time_ms(3) == pytest.approx(100.0)
    assert clock.time_ms_to_step_float(100.0) == pytest.approx(3.0)
    assert clock.latency_ms_to_frames_float(100.0) == pytest.approx(3.0)


def test_clock_rate_helpers_validate_observation_stride():
    validate_clock_rates(env_fps=60, obs_fps=10)

    assert compute_obs_stride_raw_frames(env_fps=60, obs_fps=10) == 6
    assert compute_obs_stride_raw_frames(env_fps=30, obs_fps=30) == 1

    with pytest.raises(ValueError, match="env_fps.*obs_fps"):
        compute_obs_stride_raw_frames(env_fps=60, obs_fps=17)


@pytest.mark.parametrize(
    ("env_fps", "obs_fps"),
    [
        (0, 10),
        (-30, 10),
        (30, 0),
        (30, -10),
    ],
)
def test_clock_rate_helpers_reject_nonpositive_rates(env_fps: float, obs_fps: float):
    with pytest.raises(ValueError):
        validate_clock_rates(env_fps=env_fps, obs_fps=obs_fps)


def test_ms_to_raw_frames_uses_ceil_boundaries():
    assert ms_to_raw_frames(0.0, env_fps=30) == 0
    assert ms_to_raw_frames(1.0, env_fps=30) == 1
    assert ms_to_raw_frames(1000.0 / 30.0, env_fps=30) == 1
    assert ms_to_raw_frames((1000.0 / 30.0) + 0.001, env_fps=30) == 2
    assert ms_to_raw_frames(100.0, env_fps=30) == 3

    with pytest.raises(ValueError, match="latency_ms"):
        ms_to_raw_frames(-1.0, env_fps=30)

    with pytest.raises(ValueError, match="env_fps"):
        ms_to_raw_frames(1.0, env_fps=0)


def test_env_clock_accepts_explicit_observation_fps_and_preserves_legacy_fps_env():
    clock = EnvClock(env_fps=60, obs_fps=15)

    assert clock.env_fps == 60
    assert clock.fps_env == 60
    assert clock.obs_fps == 15
    assert clock.obs_stride_raw_frames == 4
    assert clock.latency_ms_to_raw_frames(1.0) == 1

    legacy = EnvClock(fps_env=30)
    assert legacy.env_fps == 30
    assert legacy.fps_env == 30
    assert legacy.obs_fps == 30
    assert legacy.obs_stride_raw_frames == 1

    positional = EnvClock(30)
    assert positional.env_fps == 30
    assert positional.obs_fps == 30


def test_env_clock_rejects_conflicting_env_fps_aliases():
    with pytest.raises(ValueError, match="fps_env.*env_fps"):
        EnvClock(fps_env=30, env_fps=60)


def test_frame_accumulator_preserves_fractional_residuals():
    accumulator = FrameAccumulator()

    assert accumulator.consume(0.4) == 0
    assert accumulator.residual_frames == pytest.approx(0.4)
    assert accumulator.consume(0.4) == 0
    assert accumulator.residual_frames == pytest.approx(0.8)
    assert accumulator.consume(0.4) == 1
    assert accumulator.residual_frames == pytest.approx(0.2)
