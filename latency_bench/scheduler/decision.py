from __future__ import annotations

from abc import ABC, abstractmethod


class DecisionScheduler(ABC):
    @abstractmethod
    def should_observe(self, env_step: int, sim_time_ms: float) -> bool:
        ...


class EveryFrameDecisionScheduler(DecisionScheduler):
    def should_observe(self, env_step: int, sim_time_ms: float) -> bool:
        return True


class FixedFrameIntervalDecisionScheduler(DecisionScheduler):
    def __init__(self, interval_frames: int):
        if interval_frames <= 0:
            raise ValueError("interval_frames must be positive")
        self.interval_frames = int(interval_frames)

    def should_observe(self, env_step: int, sim_time_ms: float) -> bool:
        return env_step % self.interval_frames == 0


class FixedFrequencyDecisionScheduler(FixedFrameIntervalDecisionScheduler):
    def __init__(self, fps_env: float, frequency_hz: float):
        if fps_env <= 0:
            raise ValueError("fps_env must be positive")
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        interval_frames = max(1, int(round(float(fps_env) / float(frequency_hz))))
        super().__init__(interval_frames=interval_frames)
        self.fps_env = float(fps_env)
        self.frequency_hz = float(frequency_hz)

