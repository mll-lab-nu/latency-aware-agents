from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np

from latency_bench.core.types import Action, ActionEvent, EpisodeMetrics, LatencyRecord, StepRecord
from latency_bench.logging.records import to_jsonable
from latency_bench.logging.trajectory_logger import TrajectoryLogger


def _old_to_jsonable_reference(value: Any) -> Any:
    if is_dataclass(value):
        return _old_to_jsonable_reference(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _old_to_jsonable_reference(value.tolist())
    if isinstance(value, dict):
        return {str(key): _old_to_jsonable_reference(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_old_to_jsonable_reference(item) for item in value]
    return value


def _representative_records() -> list[Any]:
    action = Action(value=np.int64(1), name="flap", is_noop=False, is_oneshot=True)
    event = ActionEvent(
        action_id=7,
        obs_id=3,
        action=action,
        obs_env_step=5,
        obs_time_ms=166.666,
        latency_ms=33.333,
        ready_time_ms=200.0,
        ready_step_float=6.0,
        obs_raw_frame=5,
        latency_raw_frames=1,
        ready_raw_frame=6,
        apply_raw_frame=6,
        applied=True,
        applied_env_step=6,
        applied_time_ms=200.0,
        metadata={
            "np_scalar": np.float32(1.25),
            "np_array": np.asarray([1, 2, 3], dtype=np.int64),
            "path": Path("checkpoint.pt"),
        },
    )
    return [
        StepRecord(
            episode_id=2,
            env_step=6,
            sim_time_ms=200.0,
            active_action=action,
            reward=np.float32(0.5),
            done=False,
            info={"event": event, "score": np.int64(4), "values": np.asarray([0.0, 1.0], dtype=np.float32)},
        ),
        event,
        LatencyRecord(
            obs_id=3,
            latency_ms=33.333,
            metadata={"episode_id": 2, "slot_id": np.int64(1)},
        ),
        EpisodeMetrics(
            episode_id=2,
            episode_return=10.0,
            survival_steps=6,
            episode_return_env=10.0,
            game_score=np.float64(3.0),
            mean_latency_ms=33.333,
            num_actions=1,
            metadata={"output_dir": Path("outputs/run")},
        ),
    ]


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


def test_to_jsonable_matches_old_generic_dataclass_conversion():
    for record in _representative_records():
        old_jsonable = _old_to_jsonable_reference(record)
        new_jsonable = to_jsonable(record)

        assert new_jsonable == old_jsonable
        assert json.dumps(new_jsonable, sort_keys=True) == json.dumps(old_jsonable, sort_keys=True)


def test_to_jsonable_dataclass_path_does_not_call_asdict(monkeypatch):
    def fail_asdict(_value):
        raise AssertionError("to_jsonable should not use dataclasses.asdict for dataclass records")

    monkeypatch.setattr("latency_bench.logging.records.asdict", fail_asdict, raising=False)

    record = _representative_records()[0]

    assert to_jsonable(record) == _old_to_jsonable_reference(record)


def _write_representative_logger_output(output_dir: Path) -> dict[str, list[str]]:
    step_record, action_event, latency_record, episode_metrics = _representative_records()
    logger = TrajectoryLogger(output_dir)
    logger.log_step(step_record)
    logger.log_action_event(action_event)
    logger.log_latency(latency_record)
    logger.log_episode_metrics(episode_metrics)
    logger.flush()
    return {
        filename: (output_dir / filename).read_text(encoding="utf-8").splitlines()
        for filename in (
            "steps.jsonl",
            "actions.jsonl",
            "latencies.jsonl",
            "episode_metrics.jsonl",
        )
    }


def test_trajectory_logger_fast_jsonl_matches_old_generic_conversion(tmp_path, monkeypatch):
    old_output_dir = tmp_path / "old"
    new_output_dir = tmp_path / "new"

    monkeypatch.setattr(
        "latency_bench.logging.trajectory_logger.to_jsonable",
        _old_to_jsonable_reference,
    )
    old_lines = _write_representative_logger_output(old_output_dir)

    monkeypatch.setattr(
        "latency_bench.logging.trajectory_logger.to_jsonable",
        to_jsonable,
    )
    new_lines = _write_representative_logger_output(new_output_dir)

    assert new_lines == old_lines
    assert {
        filename: [json.loads(line) for line in lines]
        for filename, lines in new_lines.items()
    } == {
        filename: [json.loads(line) for line in lines]
        for filename, lines in old_lines.items()
    }
