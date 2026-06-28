from __future__ import annotations

import pytest

from latency_bench.latency.env_fps_calibration import select_env_fps_from_latency_samples


def test_select_env_fps_maximizes_target_frame_hits_and_preserves_obs_stride():
    calibration = select_env_fps_from_latency_samples(
        latency_samples_ms=[480.0, 510.0, 520.0],
        target_latency_frames=2,
        base_env_fps=30.0,
        base_obs_fps=15.0,
        device_count=4,
        worker_utilization_cap=1.0,
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
    calibration = select_env_fps_from_latency_samples(
        latency_samples_ms=[520.0, 530.0],
        target_latency_frames=2,
        base_env_fps=30.0,
        base_obs_fps=30.0,
        device_count=4,
        worker_utilization_cap=1.0,
    )

    assert calibration.selected_env_fps == 3
    assert calibration.hit_count == 2
    assert calibration.mean_abs_frame_error == pytest.approx(0.425)
    assert calibration.ideal_env_fps == pytest.approx(2000.0 / 525.0)


def test_select_env_fps_uses_continuous_effective_delay_not_quantized_frame_delay():
    effective_delay_calibration = select_env_fps_from_latency_samples(
        latency_samples_ms=[88.0 for _ in range(8)],
        target_latency_frames=2,
        base_env_fps=15.0,
        base_obs_fps=15.0,
        device_count=2,
        worker_utilization_cap=0.8,
    )
    quantized_frame_delay_calibration = select_env_fps_from_latency_samples(
        latency_samples_ms=[133.33333333333334 for _ in range(8)],
        target_latency_frames=2,
        base_env_fps=15.0,
        base_obs_fps=15.0,
        device_count=2,
        worker_utilization_cap=0.8,
    )

    assert effective_delay_calibration.selected_env_fps == 18
    assert quantized_frame_delay_calibration.selected_env_fps == 12


def test_select_env_fps_from_latency_samples_replaces_warmup_named_api():
    calibration = select_env_fps_from_latency_samples(
        latency_samples_ms=[34.0 for _ in range(8)],
        target_latency_frames=2,
        base_env_fps=16.0,
        base_obs_fps=16.0,
        device_count=2,
        worker_utilization_cap=1.0,
    )

    assert calibration.selected_env_fps == 58
    assert calibration.selected_obs_fps == pytest.approx(58.0)
    assert calibration.mean_latency_ms == pytest.approx(34.0)
