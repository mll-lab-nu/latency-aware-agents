from __future__ import annotations

import pytest

from latency_bench.analysis.error_metrics import (
    absolute_error,
    default_latency_equivalence_thresholds,
    latency_distribution_match_metrics,
    length_error_metrics,
    normalized_error,
    quantile_errors,
    return_error_metrics,
    tail_crossing_probability_errors,
    wasserstein_1d,
    wasserstein_frame,
)


def test_wasserstein_1d_identical_samples_is_zero():
    assert wasserstein_1d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_wasserstein_1d_shifted_equal_length_samples():
    assert wasserstein_1d([0.0, 1.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert wasserstein_1d([0.0, 0.0], [10.0, 10.0]) == pytest.approx(10.0)


def test_wasserstein_1d_unequal_sample_lengths():
    assert wasserstein_1d([0.0], [0.0, 2.0]) == pytest.approx(1.0)


def test_wasserstein_1d_rejects_empty_inputs():
    with pytest.raises(ValueError, match="samples_a must not be empty"):
        wasserstein_1d([], [1.0])
    with pytest.raises(ValueError, match="samples_b must not be empty"):
        wasserstein_1d([1.0], [])


def test_wasserstein_frame_normalizes_by_frame_ms():
    assert wasserstein_frame([0.0, 1.0], [1.0, 2.0], fps=1000.0) == pytest.approx(1.0)


def test_wasserstein_frame_rejects_invalid_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        wasserstein_frame([0.0], [1.0], fps=0.0)


def test_quantile_errors_identical_samples_are_zero():
    errors = quantile_errors([0.0, 10.0, 20.0], [0.0, 10.0, 20.0], fps=100.0)

    assert errors["p50_error_ms"] == pytest.approx(0.0)
    assert errors["p50_error_frames"] == pytest.approx(0.0)
    assert errors["p90_error_ms"] == pytest.approx(0.0)
    assert errors["p90_error_frames"] == pytest.approx(0.0)
    assert errors["p95_error_ms"] == pytest.approx(0.0)
    assert errors["p95_error_frames"] == pytest.approx(0.0)
    assert errors["p99_error_ms"] == pytest.approx(0.0)
    assert errors["p99_error_frames"] == pytest.approx(0.0)
    assert errors["max_quantile_error_frames"] == pytest.approx(0.0)


def test_quantile_errors_use_linear_percentiles():
    errors = quantile_errors([0.0, 10.0, 20.0], [10.0, 20.0, 30.0], fps=100.0)

    assert errors["p50_error_ms"] == pytest.approx(10.0)
    assert errors["p50_error_frames"] == pytest.approx(1.0)
    assert errors["p90_error_ms"] == pytest.approx(10.0)
    assert errors["p90_error_frames"] == pytest.approx(1.0)
    assert errors["p95_error_ms"] == pytest.approx(10.0)
    assert errors["p95_error_frames"] == pytest.approx(1.0)
    assert errors["p99_error_ms"] == pytest.approx(10.0)
    assert errors["p99_error_frames"] == pytest.approx(1.0)
    assert errors["max_quantile_error_frames"] == pytest.approx(1.0)


def test_quantile_errors_reject_invalid_inputs():
    with pytest.raises(ValueError, match="fps must be positive"):
        quantile_errors([1.0], [2.0], fps=-1.0)
    with pytest.raises(ValueError, match="quantiles must be in \\[0, 100\\]"):
        quantile_errors([1.0], [2.0], fps=1.0, quantiles=(-1.0,))


def test_tail_crossing_probability_errors_use_strict_greater_than():
    errors = tail_crossing_probability_errors([10.0, 50.0, 90.0], [10.0, 10.0, 90.0], fps=25.0)

    assert errors["prob_a_gt_1_frame"] == pytest.approx(2.0 / 3.0)
    assert errors["prob_b_gt_1_frame"] == pytest.approx(1.0 / 3.0)
    assert errors["tail_error_gt_1_frame"] == pytest.approx(1.0 / 3.0)
    assert errors["prob_a_gt_2_frames"] == pytest.approx(1.0 / 3.0)
    assert errors["prob_b_gt_2_frames"] == pytest.approx(1.0 / 3.0)
    assert errors["tail_error_gt_2_frames"] == pytest.approx(0.0)
    assert errors["prob_a_gt_3_frames"] == pytest.approx(0.0)
    assert errors["prob_b_gt_3_frames"] == pytest.approx(0.0)
    assert errors["tail_error_gt_3_frames"] == pytest.approx(0.0)
    assert errors["max_tail_error"] == pytest.approx(1.0 / 3.0)
def test_absolute_and_normalized_errors_with_task_scale():
    assert absolute_error(110.0, 100.0) == pytest.approx(10.0)

    metrics = normalized_error(110.0, 100.0, scale=200.0)

    assert metrics["abs_error"] == pytest.approx(10.0)
    assert metrics["norm_error"] == pytest.approx(0.05)
    assert metrics["denominator"] == pytest.approx(200.0)
    assert metrics["denominator_mode"] == "task_scale"


def test_normalized_error_realtime_denominator_fallback():
    metrics = normalized_error(110.0, 100.0, fallback_to_realtime_denominator=True)

    assert metrics["abs_error"] == pytest.approx(10.0)
    assert metrics["norm_error"] == pytest.approx(10.0 / (100.0 + 1e-8))
    assert metrics["denominator"] == pytest.approx(100.0 + 1e-8)
    assert metrics["denominator_mode"] == "realtime_denominator"


def test_normalized_error_requires_scale_or_explicit_fallback():
    with pytest.raises(ValueError, match="scale is required"):
        normalized_error(110.0, 100.0)


def test_return_and_length_error_wrappers_rename_keys():
    return_metrics = return_error_metrics(110.0, 100.0, return_scale=200.0)
    length_metrics = length_error_metrics(55.0, 50.0, length_scale=100.0)

    assert return_metrics["abs_return_error"] == pytest.approx(10.0)
    assert return_metrics["norm_return_error"] == pytest.approx(0.05)
    assert return_metrics["return_denominator"] == pytest.approx(200.0)
    assert return_metrics["return_denominator_mode"] == "task_scale"
    assert length_metrics["abs_length_error"] == pytest.approx(5.0)
    assert length_metrics["norm_length_error"] == pytest.approx(0.05)
    assert length_metrics["length_denominator"] == pytest.approx(100.0)
    assert length_metrics["length_denominator_mode"] == "task_scale"


def test_latency_distribution_match_metrics_combines_latency_metrics_without_action_age():
    metrics = latency_distribution_match_metrics([0.0, 10.0, 20.0], [10.0, 20.0, 30.0], fps=100.0)

    assert metrics["w1_ms"] == pytest.approx(10.0)
    assert metrics["w1_frames"] == pytest.approx(1.0)
    assert metrics["p50_error_frames"] == pytest.approx(1.0)
    assert metrics["tail_error_gt_1_frame"] == pytest.approx(1.0 / 3.0)
    assert "action_age_tv" not in metrics


def test_default_latency_equivalence_thresholds():
    thresholds = default_latency_equivalence_thresholds()

    assert thresholds["w1_frames"] == pytest.approx(0.1)
    assert thresholds["p50_p90_p95_error_frames"] == pytest.approx(0.25)
    assert thresholds["p99_error_frames"] == pytest.approx(0.5)
    assert thresholds["max_tail_error"] == pytest.approx(0.05)
    assert thresholds["norm_return_error"] == pytest.approx(0.1)
