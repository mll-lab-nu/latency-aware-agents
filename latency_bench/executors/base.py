"""Common executor interface.

Executors split into two families that share the same lifecycle contract
(an optional trajectory logger and ``close()``):

- :class:`EpisodeExecutor` runs one episode at a time via ``run_episode``.
- :class:`BatchedExecutor` runs many episodes concurrently via ``run_episodes``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from latency_bench.core.types import EpisodeMetrics, Observation, PolicyOutput, StepRecord
from latency_bench.logging.trajectory_logger import TrajectoryLogger


class CollectionObserver(Protocol):
    """Receive executor observations, predictions and steps for data collection."""

    capture_rgb: bool

    @property
    def stop_requested(self) -> bool:
        ...

    def start_episode(self, episode_id: int, seed: int | None, slot_id: int) -> None:
        ...

    def observation(
        self,
        episode_id: int,
        *,
        obs_id: int,
        observation: Observation,
        image: Any = None,
        latency_ms: float | None = None,
        latency_raw_frames: int | None = None,
    ) -> None:
        ...

    def output(self, episode_id: int, *, obs_id: int, output: PolicyOutput) -> None:
        ...

    def step(self, record: StepRecord, *, terminated: bool, truncated: bool) -> None:
        ...


class Executor(ABC):
    """Base for all latency executors."""
    logger: TrajectoryLogger | None

    def run_warmup(self, *, artifact_dir: str | Path | None = None) -> dict[str, Any]:
        """Warm up before evaluation; returns metadata merged into episode metrics."""
        return {}

    @abstractmethod
    def close(self) -> None:
        """Release resources owned by this executor."""
        ...


class EpisodeExecutor(Executor):
    """Executors that run a single episode per call."""
    env: Any

    def close(self) -> None:
        self.env.close()

    @abstractmethod
    def run_episode(
        self,
        episode_id: int,
        seed: int | None = None,
        eval_max_steps: int = 10000,
    ) -> EpisodeMetrics:
        ...


class BatchedExecutor(Executor):
    """Executors that run multiple episodes concurrently across slots."""
    @abstractmethod
    def run_episodes(
        self,
        *,
        episode_ids: Sequence[int],
        seeds: Sequence[int | None],
        eval_max_steps: int = 10000,
        on_episode_complete: Callable[[EpisodeMetrics], None] | None = None,
        next_episode: Callable[[], dict[str, Any] | None] | None = None,
    ) -> list[EpisodeMetrics]:
        ...
