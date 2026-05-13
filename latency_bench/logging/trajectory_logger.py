from __future__ import annotations

import json
from pathlib import Path

from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, StepRecord
from latency_bench.logging.records import to_jsonable


class TrajectoryLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_records: list[StepRecord] = []
        self.action_records: list[ActionEvent] = []
        self.latency_records: list[LatencyRecord] = []
        self.episode_metrics: list[EpisodeMetrics] = []

    def log_step(self, record: StepRecord) -> None:
        self.step_records.append(record)

    def log_action_event(self, event: ActionEvent) -> None:
        self.action_records.append(event)

    def log_latency(self, record: LatencyRecord) -> None:
        self.latency_records.append(record)

    def log_episode_metrics(self, metrics: EpisodeMetrics) -> None:
        self.episode_metrics.append(metrics)

    def flush(self) -> None:
        self._write_jsonl("steps.jsonl", self.step_records)
        self._write_jsonl("actions.jsonl", self.action_records)
        self._write_jsonl("latencies.jsonl", self.latency_records)
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
