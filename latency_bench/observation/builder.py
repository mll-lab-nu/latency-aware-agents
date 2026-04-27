from __future__ import annotations

from latency_bench.core.types import Observation
from latency_bench.envs.base import EnvAdapter
from latency_bench.observation.history import HistoryBuffer


class ObservationBuilder:
    def build(self, env: EnvAdapter, history: HistoryBuffer) -> Observation:
        latest = history.last()
        if latest is not None:
            return latest

        info = env.get_state_info()
        frame = env.render_frame()
        data = frame if frame is not None else info
        return Observation(
            data=data,
            env_step=int(info.get("env_step", 0)),
            sim_time_ms=float(info.get("sim_time_ms", 0.0)),
            metadata=info,
        )

