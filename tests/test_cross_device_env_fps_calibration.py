from __future__ import annotations

import pytest

from latency_bench.latency.env_fps_calibration import select_env_fps_from_warmup


def test_select_env_fps_maximizes_target_frame_hits_and_preserves_obs_stride():
    calibration = select_env_fps_from_warmup(
        latency_samples_ms=[480.0, 510.0, 520.0],
        target_latency_frames=2,
        base_env_fps=30.0,
        base_obs_fps=15.0,
        device_count=4,
    )

    assert calibration.selected_env_fps == 3
    assert calibration.selected_obs_fps == pytest.approx(1.5)
    assert calibration.observation_stride == pytest.approx(2.0)
    assert calibration.target_latency_frames == 2
    assert calibration.hit_count == 3
    assert calibration.sample_count == 3
    assert calibration.device_count == 4
    assert calibration.max_env_fps == 15


def test_select_env_fps_tiebreaks_by_mean_error_then_target_formula_then_lower_fps():
    calibration = select_env_fps_from_warmup(
        latency_samples_ms=[520.0, 530.0],
        target_latency_frames=2,
        base_env_fps=30.0,
        base_obs_fps=30.0,
        device_count=4,
    )

    assert calibration.selected_env_fps == 3
    assert calibration.hit_count == 2
    assert calibration.mean_abs_frame_error == pytest.approx(0.425)
    assert calibration.ideal_env_fps == pytest.approx(2000.0 / 525.0)
