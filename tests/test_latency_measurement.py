from __future__ import annotations

import pytest

from latency_bench.latency.measurement import LatencyProfiler as MeasurementLatencyProfiler
from latency_bench.latency.measurement import RuntimeLatencyMeasurer


def test_runtime_latency_measurer_records_callable_duration():
    measurer = RuntimeLatencyMeasurer(device="cpu")

    result, record = measurer.run_and_measure(lambda value: value + 1, 2, obs_id=7)

    assert result == 3
    assert record.obs_id == 7
    assert record.source == "measured"
    assert record.device == "cpu"
    assert record.latency_ms >= 0.0
    assert record.inference_ms == pytest.approx(record.latency_ms)
    assert record.end_to_end_ms == pytest.approx(record.latency_ms)
    assert record.end_wall_time >= record.start_wall_time
