from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, StepRecord
from latency_bench.logging.latency_traces import e2e_latency_rows
from latency_bench.utils.io import write_json, write_jsonl
from latency_bench.utils.stats import percentile_linear

# Realtime warmup is intentionally not configurable from YAML. The user-facing
# contract is simply "run until stable"; the numbers below define "stable" as
# stable center trend across two 50-sample windows (100 samples total). Median
# and 10% trimmed mean catch warmup drift without treating real steady-state
# tail jitter as a reason to keep warming up. p90 remains diagnostic and is
# still plotted/logged, but it does not block formal eval. Warmup is capped at
# 100 samples: a single short sub-episode is enough to warm the GPU, and a
# non-stabilizing warmup is now a non-fatal warning (see realtime.run_warmup),
# so it never aborts a multi-episode measure run.
MIN_WARMUP_LATENCY_SAMPLES = 100
MAX_WARMUP_LATENCY_SAMPLES = 100
WARMUP_EPISODE_MAX_STEPS = 150
WARMUP_WINDOW_SIZE = 40
STABLE_WINDOW_COUNT = 2
MEDIAN_REL_TOL = 0.03
TRIMMED_MEAN_REL_TOL = 0.03
MEDIAN_ABS_TOL_MS = 2.5
TRIMMED_MEAN_ABS_TOL_MS = 2.5
TRIMMED_MEAN_FRACTION = 0.10


@dataclass
class RealtimeWarmupStatus:
    stable: bool
    sample_count: int
    warmup_stop_sample: int | None
    median_span_ms: float | None
    trimmed_mean_span_ms: float | None
    p90_span_ms: float | None
    median_tolerance_ms: float | None
    trimmed_mean_tolerance_ms: float | None
    median_rel_span: float | None
    trimmed_mean_rel_span: float | None
    p90_rel_span: float | None
    reason: str


class RealtimeWarmupCollector:
    def __init__(
        self,
        *,
        save_step_records: bool = False,
        save_action_records: bool = False,
        save_latency_records: bool = True,
    ):
        self.save_step_records = save_step_records
        self.save_action_records = save_action_records
        self.save_latency_records = save_latency_records
        self.step_records: list[StepRecord] = []
        self.action_records: list[ActionEvent] = []
        self.latency_records: list[LatencyRecord] = []
        self.episode_metrics: list[EpisodeMetrics] = []

    @property
    def latency_values_ms(self) -> list[float]:
        return [float(record.latency_ms) for record in self.latency_records]

    @property
    def e2e_latency_values_ms(self) -> list[float]:
        return [row["latency_ms"] for row in e2e_latency_rows(self.action_records)]

    def log_step(self, record: StepRecord) -> None:
        if self.save_step_records:
            self.step_records.append(record)

    def log_action_event(self, event: ActionEvent) -> None:
        if self.save_action_records:
            self.action_records.append(event)

    def log_latency(self, record: LatencyRecord) -> None:
        if self.save_latency_records:
            self.latency_records.append(record)

    def log_episode_metrics(self, metrics: EpisodeMetrics) -> None:
        self.episode_metrics.append(metrics)

    def flush(self) -> None:
        pass


def is_realtime_warmup_stable(latency_values_ms: Sequence[float]) -> bool:
    return realtime_warmup_status(latency_values_ms).stable


def realtime_warmup_status(latency_values_ms: Sequence[float]) -> RealtimeWarmupStatus:
    values = [float(value) for value in latency_values_ms]
    sample_count = len(values)
    required_tail = WARMUP_WINDOW_SIZE * STABLE_WINDOW_COUNT
    if sample_count < MIN_WARMUP_LATENCY_SAMPLES or sample_count < required_tail:
        return RealtimeWarmupStatus(
            stable=False,
            sample_count=sample_count,
            warmup_stop_sample=None,
            median_span_ms=None,
            trimmed_mean_span_ms=None,
            p90_span_ms=None,
            median_tolerance_ms=None,
            trimmed_mean_tolerance_ms=None,
            median_rel_span=None,
            trimmed_mean_rel_span=None,
            p90_rel_span=None,
            reason="collecting",
        )

    tail = values[-required_tail:]
    windows = [
        tail[start:start + WARMUP_WINDOW_SIZE]
        for start in range(0, required_tail, WARMUP_WINDOW_SIZE)
    ]
    medians = [percentile_linear(window, 50.0) for window in windows]
    trimmed_means = [_trimmed_mean(window, TRIMMED_MEAN_FRACTION) for window in windows]
    p90s = [percentile_linear(window, 90.0) for window in windows]
    median_span_ms = max(medians) - min(medians)
    trimmed_mean_span_ms = max(trimmed_means) - min(trimmed_means)
    p90_span_ms = max(p90s) - min(p90s)
    median_rel_span = _relative_span(medians)
    trimmed_mean_rel_span = _relative_span(trimmed_means)
    p90_rel_span = _relative_span(p90s)
    median_tolerance_ms = max(MEDIAN_ABS_TOL_MS, abs(percentile_linear(medians, 50.0)) * MEDIAN_REL_TOL)
    trimmed_mean_tolerance_ms = max(
        TRIMMED_MEAN_ABS_TOL_MS,
        abs(percentile_linear(trimmed_means, 50.0)) * TRIMMED_MEAN_REL_TOL,
    )
    stable = (
        median_span_ms <= median_tolerance_ms
        and trimmed_mean_span_ms <= trimmed_mean_tolerance_ms
    )
    return RealtimeWarmupStatus(
        stable=stable,
        sample_count=sample_count,
        warmup_stop_sample=sample_count if stable else None,
        median_span_ms=median_span_ms,
        trimmed_mean_span_ms=trimmed_mean_span_ms,
        p90_span_ms=p90_span_ms,
        median_tolerance_ms=median_tolerance_ms,
        trimmed_mean_tolerance_ms=trimmed_mean_tolerance_ms,
        median_rel_span=median_rel_span,
        trimmed_mean_rel_span=trimmed_mean_rel_span,
        p90_rel_span=p90_rel_span,
        reason="stable" if stable else "drifting",
    )


def write_realtime_warmup_artifacts(
    output_dir: str | Path,
    collector: RealtimeWarmupCollector,
    *,
    stable: bool,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    latency_values = collector.e2e_latency_values_ms
    status = realtime_warmup_status(latency_values)
    status.stable = stable
    status.reason = "stable" if stable else status.reason

    write_jsonl(output_path / "warmup_latencies.jsonl", collector.latency_records)
    write_jsonl(output_path / "warmup_infer_latencies.jsonl", collector.latency_records)
    write_jsonl(output_path / "warmup_e2e_latencies.jsonl", e2e_latency_rows(collector.action_records))
    write_json(output_path / "warmup_summary.json", _summary_payload(status, latency_values))
    plot_realtime_warmup(latency_values, output_path / "warmup_stability.png", status=status)
    return _summary_payload(status, latency_values)


def plot_realtime_warmup(
    latency_values_ms: Sequence[float],
    output_path: str | Path,
    *,
    status: RealtimeWarmupStatus | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [float(value) for value in latency_values_ms]
    status = status or realtime_warmup_status(values)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(range(1, len(values) + 1), values, linewidth=1.2, label="latency")
    if len(values) >= WARMUP_WINDOW_SIZE:
        xs, rolling = _rolling_median(values, WARMUP_WINDOW_SIZE)
        ax.plot(xs, rolling, linewidth=2.0, label=f"rolling median ({WARMUP_WINDOW_SIZE})")
        p90_xs, rolling_p90 = _rolling_percentile(values, WARMUP_WINDOW_SIZE, 90.0)
        ax.plot(p90_xs, rolling_p90, linewidth=1.6, label=f"rolling p90 ({WARMUP_WINDOW_SIZE})")
    ax.axvline(MIN_WARMUP_LATENCY_SAMPLES, color="0.4", linestyle="--", linewidth=1.0, label="min samples")
    _shade_warmup_windows(ax, status=status)
    if status.warmup_stop_sample is not None:
        ax.axvline(
            status.warmup_stop_sample,
            color="0.1",
            linestyle="-.",
            linewidth=1.2,
            label="warmup stop",
        )
    ax.set_title(f"Realtime warmup: {status.reason}")
    ax.set_xlabel("warmup latency sample")
    ax.set_ylabel("latency (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_realtime_eval_latency(
    latency_values_ms: Sequence[float],
    output_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [float(value) for value in latency_values_ms]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(range(1, len(values) + 1), values, linewidth=1.2, label="formal eval latency")
    if len(values) >= WARMUP_WINDOW_SIZE:
        xs, rolling = _rolling_median(values, WARMUP_WINDOW_SIZE)
        ax.plot(xs, rolling, linewidth=2.0, label=f"rolling median ({WARMUP_WINDOW_SIZE})")
    ax.set_title("Realtime formal eval latency")
    ax.set_xlabel("formal eval latency sample")
    ax.set_ylabel("latency (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _summary_payload(status: RealtimeWarmupStatus, latency_values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in latency_values]
    return {
        "warmup_status": "stable" if status.stable else "unstable",
        "warmup_samples": len(values),
        "warmup_stop_sample": status.warmup_stop_sample,
        "warmup_mean_latency_ms": sum(values) / len(values) if values else None,
        "warmup_p50_latency_ms": percentile_linear(values, 50.0) if values else None,
        "warmup_p90_latency_ms": percentile_linear(values, 90.0) if values else None,
        "warmup_median_span_ms": status.median_span_ms,
        "warmup_trimmed_mean_span_ms": status.trimmed_mean_span_ms,
        "warmup_p90_span_ms": status.p90_span_ms,
        "warmup_median_tolerance_ms": status.median_tolerance_ms,
        "warmup_trimmed_mean_tolerance_ms": status.trimmed_mean_tolerance_ms,
        "warmup_median_rel_span": status.median_rel_span,
        "warmup_trimmed_mean_rel_span": status.trimmed_mean_rel_span,
        "warmup_p90_rel_span": status.p90_rel_span,
        "warmup_stability_reason": status.reason,
    }


def _relative_span(values: Sequence[float]) -> float:
    center = percentile_linear(values, 50.0)
    span = max(values) - min(values)
    if center == 0.0:
        return 0.0 if span == 0.0 else math.inf
    return span / abs(center)


def _trimmed_mean(values: Sequence[float], trim_fraction: float) -> float:
    sorted_values = sorted(float(value) for value in values)
    trim_count = int(len(sorted_values) * trim_fraction)
    trimmed = sorted_values[trim_count:len(sorted_values) - trim_count]
    return sum(trimmed) / len(trimmed)


def _rolling_median(values: Sequence[float], window_size: int) -> tuple[list[int], list[float]]:
    return _rolling_percentile(values, window_size, 50.0)


def _rolling_percentile(
    values: Sequence[float],
    window_size: int,
    percentile: float,
) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    percentiles: list[float] = []
    for end in range(window_size, len(values) + 1):
        xs.append(end)
        percentiles.append(percentile_linear(values[end - window_size:end], percentile))
    return xs, percentiles


def _shade_warmup_windows(ax: Any, *, status: RealtimeWarmupStatus) -> None:
    end = status.warmup_stop_sample or status.sample_count
    required_tail = WARMUP_WINDOW_SIZE * STABLE_WINDOW_COUNT
    if end < required_tail:
        return
    start = end - required_tail
    for window_idx in range(STABLE_WINDOW_COUNT):
        window_start = start + window_idx * WARMUP_WINDOW_SIZE + 1
        window_end = start + (window_idx + 1) * WARMUP_WINDOW_SIZE
        ax.axvspan(
            window_start,
            window_end,
            color="0.85",
            alpha=0.18 if window_idx % 2 == 0 else 0.28,
            linewidth=0,
        )
