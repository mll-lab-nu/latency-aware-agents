from __future__ import annotations

from latency_bench.executors.realtime_warmup import (
    is_realtime_warmup_stable,
    plot_realtime_warmup,
    realtime_warmup_status,
)


def _stable_mock_latencies() -> list[float]:
    return [
        20.0 + (idx % 5) * 0.04
        for idx in range(300)
    ]


def _unstable_mock_latencies() -> list[float]:
    # Steady downward drift; across the trailing 100-sample stability window
    # (two 50-sample windows) the window-center span exceeds tolerance, so
    # warmup is judged not-yet-stable.
    return [
        30.0 - idx * 0.1
        for idx in range(120)
    ]


def _stable_center_with_tail_jitter() -> list[float]:
    # Stable ~86ms center with occasional high tail spikes. Across two 50-sample
    # windows the trimmed mean stays within tolerance (spikes get trimmed) while
    # p90 still reflects the spikes, so warmup is stable but p90 span is larger.
    samples: list[float] = []
    for spike in (95.0, 115.0):
        samples.extend(86.0 + (idx % 5) * 0.08 for idx in range(44))
        samples.extend(spike for _ in range(6))
    return samples


def test_realtime_warmup_stability_accepts_stable_mock_data(tmp_path):
    samples = _stable_mock_latencies()
    status = realtime_warmup_status(samples)

    assert status.stable is True
    assert is_realtime_warmup_stable(samples) is True

    path = tmp_path / "mock_stable_warmup.png"
    plot_realtime_warmup(samples, path, status=status)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_realtime_warmup_stability_accepts_stable_center_with_tail_jitter(tmp_path):
    samples = _stable_center_with_tail_jitter()
    status = realtime_warmup_status(samples)

    assert status.stable is True
    assert status.p90_span_ms > status.median_span_ms
    assert status.trimmed_mean_span_ms <= status.trimmed_mean_tolerance_ms

    path = tmp_path / "mock_tail_jitter_warmup.png"
    plot_realtime_warmup(samples, path, status=status)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_realtime_warmup_stability_rejects_unstable_mock_data(tmp_path):
    samples = _unstable_mock_latencies()
    status = realtime_warmup_status(samples)

    assert status.stable is False
    assert is_realtime_warmup_stable(samples) is False

    path = tmp_path / "mock_unstable_warmup.png"
    plot_realtime_warmup(samples, path, status=status)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_realtime_warmup_stability_accepts_zero_latency_data(tmp_path):
    samples = [0.0 for _ in range(300)]
    status = realtime_warmup_status(samples)

    assert status.stable is True
    assert status.median_rel_span == 0.0
    assert status.p90_rel_span == 0.0

    path = tmp_path / "zero_latency_warmup.png"
    plot_realtime_warmup(samples, path, status=status)

    assert path.is_file()
    assert path.stat().st_size > 0
