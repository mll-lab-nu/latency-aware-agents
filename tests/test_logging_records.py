from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from latency_bench.core.types import Action, ActionEvent, EpisodeMetrics, LatencyRecord, StepRecord
from latency_bench.logging.latency_traces import e2e_latency_rows
from latency_bench.logging.records import step_record_info, to_jsonable
from latency_bench.logging.trajectory_logger import TrajectoryLogger


def _realtime_profile(bench_overhead_ms: float = 0.0) -> dict[str, float]:
    return {
        "env_observe_ms": 0.0,
        "env_to_parent_ipc_ms": 0.0,
        "parent_observation_handle_ms": 0.0,
        "parent_to_worker_ipc_ms": 0.0,
        "worker_recv_overhead_ms": 0.0,
        "worker_to_parent_ipc_ms": 0.0,
        "parent_release_ms": 0.0,
        "parent_to_env_ipc_ms": bench_overhead_ms,
        "env_action_recv_to_enqueue_ms": 0.0,
        "env_scheduler_update_ms": 0.0,
        "env_step_ms": 0.0,
        "env_wall_lag_ms": 0.0,
    }


def _trajectory_logger(
    output_dir: Path,
    *,
    save_step_records: bool = True,
    save_action_records: bool = True,
    save_latency_records: bool = True,
) -> TrajectoryLogger:
    return TrajectoryLogger(
        output_dir,
        save_step_records=save_step_records,
        save_action_records=save_action_records,
        save_latency_records=save_latency_records,
    )


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
        action=action,
        obs_env_step=5,
        ready_time_ms=200.0,
        ready_step_float=6.0,
        latency_raw_frames=1,
        ready_raw_frame=6,
        apply_raw_frame=6,
        applied=True,
        latency=LatencyRecord(
            obs_id=3,
            latency_ms=33.333,
        ),
        metadata={
            "frame_ms": 33.333,
            "realtime_profile": _realtime_profile(2.0),
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
    logger = _trajectory_logger(tmp_path)
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


def test_step_record_info_omits_raw_rgb_frame_stack():
    info = {
        "env_raw_rgb_frame_stack": np.zeros((4, 84, 84, 3), dtype=np.uint8),
        "score": 2,
        "active_action_id": 3,
    }

    compact_info = step_record_info(info)

    assert "env_raw_rgb_frame_stack" not in compact_info
    assert compact_info == {"active_action_id": 3, "score": 2}


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
    logger = _trajectory_logger(output_dir)
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
        "latency_bench.utils.io.to_jsonable",
        _old_to_jsonable_reference,
    )
    old_lines = _write_representative_logger_output(old_output_dir)

    monkeypatch.setattr(
        "latency_bench.utils.io.to_jsonable",
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


def test_trajectory_logger_writes_inference_and_e2e_latency_traces(tmp_path):
    logger = _trajectory_logger(tmp_path)
    event = ActionEvent(
        action_id=3,
        action=Action("jump"),
        obs_env_step=2,
        ready_time_ms=75.0,
        ready_step_float=3.0,
        latency=LatencyRecord(obs_id=2, latency_ms=75.0, device="cuda:0"),
        latency_raw_frames=2,
        ready_raw_frame=4,
        apply_raw_frame=5,
        applied=True,
        metadata={"frame_ms": 100.0 / 3.0, "realtime_profile": _realtime_profile(8.5)},
    )
    logger.log_action_event(event)
    logger.log_latency(event.latency)
    logger.flush()

    inference_rows = [
        json.loads(line)
        for line in (tmp_path / "infer_latencies.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    e2e_rows = [
        json.loads(line)
        for line in (tmp_path / "e2e_latencies.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert inference_rows[0]["latency_ms"] == 75.0
    assert e2e_rows == [
        {
            "action_id": 3,
            "applied_latency_frames": 3,
            "applied_latency_ms": 100.0,
            "apply_raw_frame": 5,
            "bench_overhead_ms": 8.5,
            "e2e_latency_ms": 83.5,
            "inference_latency_ms": 75.0,
            "latency_ms": 83.5,
            "obs_env_step": 2,
        }
    ]


def test_e2e_latency_rows_falls_back_to_frame_delay_without_profile():
    # Profiling disabled is a legitimate state: no profile means fall back to
    # the frame-quantized applied delay rather than raising.
    event = ActionEvent(
        action_id=3,
        action=Action("jump"),
        obs_env_step=2,
        ready_time_ms=75.0,
        ready_step_float=3.0,
        latency=LatencyRecord(obs_id=2, latency_ms=75.0, device="cuda:0"),
        latency_raw_frames=2,
        ready_raw_frame=4,
        apply_raw_frame=5,
        applied=True,
        metadata={"frame_ms": 100.0 / 3.0},
    )

    rows = e2e_latency_rows([event])

    assert rows == [
        {
            "action_id": 3,
            "applied_latency_frames": 3,
            "applied_latency_ms": 100.0,
            "apply_raw_frame": 5,
            "e2e_latency_ms": 100.0,
            "inference_latency_ms": 75.0,
            "latency_ms": 100.0,
            "obs_env_step": 2,
        }
    ]
    assert "bench_overhead_ms" not in rows[0]


def test_e2e_latency_rows_raises_on_partial_profile():
    # A profile that is present but incomplete is a bug, not a disabled run:
    # fail fast instead of silently treating missing overhead stages as zero.
    partial = _realtime_profile(8.5)
    del partial["parent_release_ms"]
    event = ActionEvent(
        action_id=3,
        action=Action("jump"),
        obs_env_step=2,
        ready_time_ms=75.0,
        ready_step_float=3.0,
        latency=LatencyRecord(obs_id=2, latency_ms=75.0, device="cuda:0"),
        apply_raw_frame=5,
        applied=True,
        metadata={"frame_ms": 100.0 / 3.0, "realtime_profile": partial},
    )

    with pytest.raises(ValueError, match="missing measured stages"):
        e2e_latency_rows([event])


def test_e2e_latency_rows_requires_frame_ms_for_applied_actions():
    event = ActionEvent(
        action_id=3,
        action=Action("jump"),
        obs_env_step=2,
        ready_time_ms=75.0,
        ready_step_float=3.0,
        latency=LatencyRecord(obs_id=2, latency_ms=75.0, device="cuda:0"),
        latency_raw_frames=2,
        ready_raw_frame=4,
        apply_raw_frame=5,
        applied=True,
        metadata={"realtime_profile": _realtime_profile(8.5)},
    )

    with pytest.raises(KeyError, match="frame_ms"):
        e2e_latency_rows([event])


def test_trajectory_logger_does_not_collect_or_write_disabled_records(tmp_path):
    step_record, action_event, latency_record, episode_metrics = _representative_records()
    logger = _trajectory_logger(
        tmp_path,
        save_step_records=False,
        save_action_records=False,
        save_latency_records=False,
    )

    logger.log_step(step_record)
    logger.log_action_event(action_event)
    logger.log_latency(latency_record)
    logger.log_episode_metrics(episode_metrics)
    logger.flush()

    assert logger.step_records == []
    assert logger.action_records == []
    assert logger.latency_records == []
    assert logger.episode_metrics == [episode_metrics]
    assert not (tmp_path / "steps.jsonl").exists()
    assert not (tmp_path / "actions.jsonl").exists()
    assert not (tmp_path / "latencies.jsonl").exists()
    assert (tmp_path / "episode_metrics.jsonl").exists()
