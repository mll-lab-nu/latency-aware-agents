"""Common executor interface.

Executors split into two families that share the same lifecycle contract
(an environment handle, an optional trajectory logger, and ``close()``):

- :class:`EpisodeExecutor` runs one episode at a time via ``run_episode``.
- :class:`BatchedExecutor` runs many episodes concurrently via ``run_episodes``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from latency_bench.core.types import EpisodeMetrics
    from latency_bench.logging.trajectory_logger import TrajectoryLogger


class Executor(ABC):
    """Base for all latency executors."""
    env: Any
    logger: TrajectoryLogger | None

    def run_warmup(self, *, artifact_dir: str | Path | None = None) -> dict[str, Any]:
        """Warm up before evaluation; returns metadata merged into episode metrics."""
        return {}

    def close(self) -> None:
        self.env.close()


class EpisodeExecutor(Executor):
    """Executors that run a single episode per call."""
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
    ) -> list[EpisodeMetrics]:
        ...
