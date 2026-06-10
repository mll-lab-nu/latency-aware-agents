from __future__ import annotations

import pytest

from latency_bench.analysis.profile_stats import compute_latency_profile_summary


def test_compute_latency_profile_summary_basic_stats():
    summary = compute_latency_profile_summary([10.0, 20.0, 30.0, 40.0], env_fps=25.0)

    assert summary["n_samples"] == 4
    assert summary["mean_ms"] == pytest.approx(25.0)
    assert summary["std_ms"] == pytest.approx(11.1803398875)
    assert summary["min_ms"] == pytest.approx(10.0)
    assert summary["max_ms"] == pytest.approx(40.0)
    assert summary["frame_ms"] == pytest.approx(40.0)
    assert summary["mean_effective_frames"] == pytest.approx(25.0 / 40.0)
    assert summary["prob_latency_gt_1_frame"] == pytest.approx(0.0)


def test_compute_latency_profile_summary_uses_linear_percentiles():
    summary = compute_latency_profile_summary([10.0, 20.0, 30.0, 40.0], env_fps=25.0)

    assert summary["p50_ms"] == pytest.approx(25.0)
    assert summary["p90_ms"] == pytest.approx(37.0)
    assert summary["p95_ms"] == pytest.approx(38.5)
    assert summary["p99_ms"] == pytest.approx(39.7)
    assert summary["p50_frames"] == pytest.approx(25.0 / 40.0)
    assert summary["p90_frames"] == pytest.approx(37.0 / 40.0)
    assert summary["p95_frames"] == pytest.approx(38.5 / 40.0)
    assert summary["p99_frames"] == pytest.approx(39.7 / 40.0)


def test_compute_latency_profile_summary_tail_probabilities():
    summary = compute_latency_profile_summary([10.0, 50.0, 90.0], env_fps=25.0)

    assert summary["prob_latency_gt_1_frame"] == pytest.approx(2.0 / 3.0)
    assert summary["prob_latency_gt_2_frames"] == pytest.approx(1.0 / 3.0)
    assert summary["prob_latency_gt_3_frames"] == pytest.approx(0.0)


def test_compute_latency_profile_summary_custom_tail_thresholds():
    summary = compute_latency_profile_summary([10.0, 50.0, 90.0], env_fps=25.0, tail_frame_thresholds=(1, 4))

    assert summary["prob_latency_gt_1_frame"] == pytest.approx(2.0 / 3.0)
    assert summary["prob_latency_gt_4_frames"] == pytest.approx(0.0)
    assert "prob_latency_gt_2_frames" not in summary


@pytest.mark.parametrize("env_fps", [0.0, -25.0])
def test_compute_latency_profile_summary_rejects_invalid_env_fps(env_fps: float):
    with pytest.raises(ValueError, match="env_fps must be positive"):
        compute_latency_profile_summary([10.0], env_fps=env_fps)


def test_compute_latency_profile_summary_rejects_empty_samples():
    with pytest.raises(ValueError, match="samples_ms must not be empty"):
        compute_latency_profile_summary([], env_fps=25.0)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_compute_latency_profile_summary_rejects_invalid_samples(value: float):
    with pytest.raises(ValueError, match="samples_ms must contain finite non-negative values"):
        compute_latency_profile_summary([value], env_fps=25.0)
