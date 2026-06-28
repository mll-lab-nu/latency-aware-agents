from __future__ import annotations

import json

import pytest

from latency_bench.latency.runtime_telemetry import (
    RuntimeTelemetryMonitor,
    parse_nvidia_smi_sample,
    summarize_runtime_telemetry,
)


def test_parse_nvidia_smi_sample_records_numeric_gpu_fields():
    row = (
        "2026/06/21 12:00:00.000, GPU-0, NVIDIA L40S, 95 %, 41 %, "
        "281.50 W, 66, P0, 2100 MHz, 9000 MHz, 4, 16"
    )

    sample = parse_nvidia_smi_sample(row)

    assert sample["timestamp"] == "2026/06/21 12:00:00.000"
    assert sample["uuid"] == "GPU-0"
    assert sample["name"] == "NVIDIA L40S"
    assert sample["utilization.gpu"] == pytest.approx(95.0)
    assert sample["utilization.memory"] == pytest.approx(41.0)
    assert sample["power.draw"] == pytest.approx(281.5)
    assert sample["temperature.gpu"] == pytest.approx(66.0)
    assert sample["pstate"] == "P0"
    assert sample["clocks.current.sm"] == pytest.approx(2100.0)
    assert sample["clocks.current.memory"] == pytest.approx(9000.0)
    assert sample["pcie.link.gen.current"] == pytest.approx(4.0)
    assert sample["pcie.link.width.current"] == pytest.approx(16.0)


def test_summarize_runtime_telemetry_writes_per_gpu_stats(tmp_path):
    telemetry = tmp_path / "runtime_telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            [
                json.dumps({"uuid": "GPU-0", "utilization.gpu": 50.0, "power.draw": 200.0, "pstate": "P0"}),
                json.dumps({"uuid": "GPU-0", "utilization.gpu": 100.0, "power.draw": 300.0, "pstate": "P0"}),
                json.dumps({"uuid": "GPU-1", "utilization.gpu": 0.0, "power.draw": 40.0, "pstate": "P8"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_runtime_telemetry(telemetry, tmp_path / "runtime_telemetry_summary.json")

    assert summary["num_samples"] == 3
    assert summary["gpus"]["GPU-0"]["sample_count"] == 2
    assert summary["gpus"]["GPU-0"]["utilization.gpu"]["mean"] == pytest.approx(75.0)
    assert summary["gpus"]["GPU-0"]["power.draw"]["max"] == pytest.approx(300.0)
    assert summary["gpus"]["GPU-0"]["pstate_counts"] == {"P0": 2}
    assert summary["gpus"]["GPU-1"]["pstate_counts"] == {"P8": 1}
    assert json.loads((tmp_path / "runtime_telemetry_summary.json").read_text(encoding="utf-8")) == summary


def test_runtime_telemetry_monitor_start_failure_is_nonfatal(tmp_path):
    monitor = RuntimeTelemetryMonitor(
        output_dir=tmp_path,
        sample_fn=lambda: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")),
    )

    status = monitor.start()
    monitor.stop()

    assert status["runtime_telemetry_enabled"] is False
    assert "nvidia-smi" in status["runtime_telemetry_error"]


def test_runtime_telemetry_monitor_writes_jsonl_samples(tmp_path):
    monitor = RuntimeTelemetryMonitor(
        output_dir=tmp_path,
        sample_fn=lambda: [{"uuid": "GPU-0", "utilization.gpu": 90.0, "pstate": "P0"}],
    )

    status = monitor.start()
    stop_status = monitor.stop()

    rows = [
        json.loads(line)
        for line in (tmp_path / "runtime_telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert status["runtime_telemetry_enabled"] is True
    assert rows == [{"uuid": "GPU-0", "utilization.gpu": 90.0, "pstate": "P0"}]
    assert stop_status["runtime_telemetry_summary_path"] == "runtime_telemetry_summary.json"
