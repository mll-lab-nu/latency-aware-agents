from __future__ import annotations

from abc import ABC, abstractmethod

from latency_bench.core.clock import EnvClock


class DecisionScheduler(ABC):
    @abstractmethod
    def should_observe(self, env_step: int, sim_time_ms: float) -> bool:
        ...


class ObservationCadenceDecisionScheduler(DecisionScheduler):
    def __init__(self, clock: EnvClock):
        self.clock = clock
        self.interval_frames = clock.obs_stride_raw_frames

    def should_observe(self, env_step: int, sim_time_ms: float) -> bool:
        del sim_time_ms
        return env_step % self.interval_frames == 0
