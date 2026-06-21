from __future__ import annotations

import json
import subprocess
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from latency_bench.utils.io import write_json
from latency_bench.utils.stats import series_stats


NVIDIA_SMI_TELEMETRY_FIELDS = (
    "timestamp",
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "temperature.gpu",
    "pstate",
    "clocks.current.sm",
    "clocks.current.memory",
    "pcie.link.gen.current",
    "pcie.link.width.current",
)

_NUMERIC_FIELDS = (
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
    "pcie.link.gen.current",
    "pcie.link.width.current",
)


def parse_nvidia_smi_sample(row: str) -> dict[str, Any]:
    values = [item.strip() for item in row.split(",")]
    sample = {field: value for field, value in zip(NVIDIA_SMI_TELEMETRY_FIELDS, values)}
    for field in _NUMERIC_FIELDS:
        sample[field] = _number(sample[field])
    return sample


def summarize_runtime_telemetry(path: str | Path, output_path: str | Path) -> dict[str, Any]:
    rows = [_load_telemetry_line(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    by_gpu: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gpu[str(row["uuid"])].append(row)
    summary = {
        "num_samples": len(rows),
        "gpus": {
            uuid: _summarize_gpu_rows(gpu_rows)
            for uuid, gpu_rows in sorted(by_gpu.items())
        },
    }
    write_json(output_path, summary)
    return summary


class RuntimeTelemetryMonitor:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        interval_s: int = 1,
        sample_fn: Callable[[], list[dict[str, Any]]] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.output_dir = Path(output_dir)
        self.interval_s = interval_s
        self.sample_fn = sample_fn or _sample_nvidia_smi
        self.sleep_fn = sleep_fn
        self.telemetry_path = self.output_dir / "runtime_telemetry.jsonl"
        self.summary_path = self.output_dir / "runtime_telemetry_summary.json"
        self._file = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    def start(self) -> dict[str, Any]:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._file = self.telemetry_path.open("w", encoding="utf-8")
            self._write_samples(self.sample_fn())
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        except Exception as exc:
            if self._file is not None:
                self._file.close()
                self._file = None
            return {
                "runtime_telemetry_enabled": False,
                "runtime_telemetry_error": str(exc),
            }
        return {
            "runtime_telemetry_enabled": True,
            "runtime_telemetry_path": self.telemetry_path.name,
        }

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._file is not None:
            self._file.close()
            self._file = None
        if self.telemetry_path.exists() and self.telemetry_path.stat().st_size > 0:
            summary = summarize_runtime_telemetry(self.telemetry_path, self.summary_path)
            result = {
                "runtime_telemetry_summary_path": self.summary_path.name,
                "runtime_telemetry_samples": summary["num_samples"],
            }
            if self._error is not None:
                result["runtime_telemetry_error"] = self._error
            return result
        return {}

    def _run_loop(self) -> None:
        while not self._stop_event.wait(float(self.interval_s)):
            try:
                self._write_samples(self.sample_fn())
            except Exception as exc:
                self._error = str(exc)
                return

    def _write_samples(self, samples: list[dict[str, Any]]) -> None:
        for sample in samples:
            self._file.write(json.dumps(sample, sort_keys=True) + "\n")
        self._file.flush()


def _summarize_gpu_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sample_count": len(rows)}
    for field in _NUMERIC_FIELDS:
        values = [float(row[field]) for row in rows if field in row]
        if values:
            summary[field] = series_stats(values)
    pstates = Counter(str(row["pstate"]) for row in rows if "pstate" in row)
    if pstates:
        summary["pstate_counts"] = dict(sorted(pstates.items()))
    return summary


def _load_telemetry_line(line: str) -> dict[str, Any]:
    if line.startswith("{"):
        return json.loads(line)
    return parse_nvidia_smi_sample(line)


def _sample_nvidia_smi() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(NVIDIA_SMI_TELEMETRY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return [parse_nvidia_smi_sample(line) for line in output.splitlines() if line]


def _number(value: str) -> float:
    return float(
        value.replace("%", "")
        .replace("W", "")
        .replace("C", "")
        .replace("MHz", "")
        .replace("MiB", "")
        .strip()
    )
