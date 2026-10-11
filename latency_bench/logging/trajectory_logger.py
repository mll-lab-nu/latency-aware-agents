from __future__ import annotations

from pathlib import Path

from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, StepRecord
from latency_bench.logging.records import e2e_latency_rows
from latency_bench.utils.io import append_jsonl, write_jsonl

_OUTPUT_FILENAMES = (
    "steps.jsonl",
    "actions.jsonl",
    "observation_attempts.jsonl",
    "e2e_latencies.jsonl",
    "latencies.jsonl",
    "episode_metrics.jsonl",
)


class TrajectoryLogger:
    """Streams per-step/per-action records to disk at episode boundaries.

    Heavy record lists are appended to their jsonl files and cleared each time
    an episode completes, so the process heap stays bounded: a run-long
    accumulation makes CPython gen-2 GC pauses grow past 100ms, which stalls
    the realtime control loop mid-episode.

    Constructing a logger claims the output dir: stale output files from a
    previous run are deleted up front, so every write is a plain append.
    """
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
        for filename in _OUTPUT_FILENAMES:
            (self.output_dir / filename).unlink(missing_ok=True)
        self.save_step_records = save_step_records
        self.save_action_records = save_action_records
        self.save_latency_records = save_latency_records
        self.step_records: list[StepRecord] = []
        self.action_records: list[ActionEvent] = []
        self.observation_attempt_records: list[dict] = []
        self.latency_records: list[LatencyRecord] = []
        self.episode_metrics: list[EpisodeMetrics] = []
        self.latency_ms_values: list[float] = []

    def log_step(self, record: StepRecord) -> None:
        if self.save_step_records:
            self.step_records.append(record)

    def log_action_event(self, event: ActionEvent) -> None:
        if self.save_action_records:
            self.action_records.append(event)

    def log_latency(self, record: LatencyRecord) -> None:
        if self.save_latency_records:
            self.latency_records.append(record)
            self.latency_ms_values.append(float(record.latency_ms))

    def log_observation_attempts(self, records) -> None:
        self.observation_attempt_records.extend(records)

    def log_episode_metrics(self, metrics: EpisodeMetrics) -> None:
        """
        Called at the end of each episode
        Flush episode into disk instead of staying in memory
        """
        self.episode_metrics.append(metrics)
        self._flush_episode_records()

    def flush(self) -> None:
        self._flush_episode_records()
        # Episode metrics stay in memory until the end: the driver backfills
        # their metadata after all episodes have run.
        write_jsonl(self.output_dir / "episode_metrics.jsonl", self.episode_metrics)

    def _flush_episode_records(self) -> None:
        append_jsonl(
            self.output_dir / "observation_attempts.jsonl",
            self.observation_attempt_records,
        )
        self.observation_attempt_records = []
        if self.save_step_records:
            append_jsonl(self.output_dir / "steps.jsonl", self.step_records)
            self.step_records = []
        if self.save_action_records:
            append_jsonl(self.output_dir / "actions.jsonl", self.action_records)
            append_jsonl(self.output_dir / "e2e_latencies.jsonl", e2e_latency_rows(self.action_records))
            self.action_records = []
        if self.save_latency_records:
            append_jsonl(self.output_dir / "latencies.jsonl", self.latency_records)
            self.latency_records = []
