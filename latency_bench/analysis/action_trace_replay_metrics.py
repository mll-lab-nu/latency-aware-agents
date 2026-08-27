from __future__ import annotations

import math
from typing import Any

from latency_bench.core.types import StepRecord


def replay_step_records(env: Any, step_records: list[StepRecord]) -> dict[str, Any]:
    episode_return = 0.0
    reward_mismatch_count = 0
    done_mismatch_count = 0
    replay_done = False
    steps_replayed = 0
    for record in step_records:
        result = env.step(record.active_action)
        replay_done = bool(result.done or result.truncated)
        episode_return += float(result.reward)
        if not math.isclose(
            float(result.reward),
            float(record.reward),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            reward_mismatch_count += 1
        if replay_done != bool(record.done):
            done_mismatch_count += 1
        steps_replayed += 1
        if replay_done:
            break
    return {
        "steps_replayed": steps_replayed,
        "episode_return": episode_return,
        "done": replay_done,
        "reward_mismatch_count": reward_mismatch_count,
        "done_mismatch_count": done_mismatch_count,
    }
