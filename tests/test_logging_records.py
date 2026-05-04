from __future__ import annotations

import json

import numpy as np

from latency_bench.core.types import Action, StepRecord
from latency_bench.logging.records import to_jsonable
from latency_bench.logging.trajectory_logger import TrajectoryLogger


def test_to_jsonable_converts_numpy_arrays_and_scalars():
    value = {
        "array": np.asarray([1, 2, 3], dtype=np.int64),
        "scalar": np.float32(1.5),
    }

    converted = to_jsonable(value)

    assert converted == {
        "array": [1, 2, 3],
        "scalar": 1.5,
    }
    json.dumps(converted)


def test_trajectory_logger_flushes_step_info_with_numpy_values(tmp_path):
    logger = TrajectoryLogger(tmp_path)
    logger.log_step(
        StepRecord(
            episode_id=0,
            env_step=1,
            sim_time_ms=33.3,
            active_action=Action(value=np.int64(1), name="flap"),
            reward=1.0,
            done=False,
            info={
                "numeric_obs": np.asarray([0.0, 1.0], dtype=np.float32),
                "score": np.int64(2),
            },
        )
    )

    logger.flush()

    rows = [
        json.loads(line)
        for line in (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["active_action"]["value"] == 1
    assert rows[0]["info"]["numeric_obs"] == [0.0, 1.0]
    assert rows[0]["info"]["score"] == 2
