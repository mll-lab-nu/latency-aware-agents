from __future__ import annotations

import json
from pathlib import Path

from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, StepRecord
from latency_bench.logging.latency_traces import e2e_latency_rows
from latency_bench.logging.records import to_jsonable


class TrajectoryLogger:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_step_records: bool,
        save_action_records: bool,
        save_latency_records: bool,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_step_records = save_step_records
        self.save_action_records = save_action_records
        self.save_latency_records = save_latency_records
        self.step_records: list[StepRecord] = []
        self.action_records: list[ActionEvent] = []
        self.latency_records: list[LatencyRecord] = []
        self.episode_metrics: list[EpisodeMetrics] = []

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
        if self.save_step_records:
            self._write_jsonl("steps.jsonl", self.step_records)
        if self.save_action_records:
            self._write_jsonl("actions.jsonl", self.action_records)
            self._write_jsonl("e2e_latencies.jsonl", e2e_latency_rows(self.action_records))
        if self.save_latency_records:
            self._write_jsonl("latencies.jsonl", self.latency_records)
            self._write_jsonl("infer_latencies.jsonl", self.latency_records)
        self._write_jsonl("episode_metrics.jsonl", self.episode_metrics)

    def _write_jsonl(self, filename: str, records: list) -> None:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            dumps = json.dumps
            convert = to_jsonable
            f.writelines(
                dumps(convert(record), sort_keys=True) + "\n"
                for record in records
            )
