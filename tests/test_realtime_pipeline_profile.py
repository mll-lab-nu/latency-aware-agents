from __future__ import annotations

import json

from latency_bench.core.types import Action, ActionEvent, LatencyRecord
from latency_bench.logging.realtime_pipeline import realtime_pipeline_profile_rows, realtime_pipeline_summary
from latency_bench.logging.trajectory_logger import TrajectoryLogger


def _profiled_event() -> ActionEvent:
    event = ActionEvent(
        action_id=3,
        action=Action("jump"),
        obs_env_step=10,
        ready_time_ms=140.0,
        ready_step_float=4.2,
        latency=LatencyRecord(obs_id=7, latency_ms=40.0, device="cuda:0"),
        apply_raw_frame=15,
        applied=True,
        metadata={
            "frame_ms": 10.0,
            "realtime_profile": {
                "env_observe_ms": 1.0,
                "env_to_parent_ipc_ms": 2.0,
                "parent_observation_handle_ms": 3.0,
                "parent_to_worker_ipc_ms": 4.0,
                "worker_recv_overhead_ms": 5.0,
                "worker_to_parent_ipc_ms": 6.0,
                "parent_release_ms": 7.0,
                "parent_to_env_ipc_ms": 8.0,
                "env_action_recv_to_enqueue_ms": 9.0,
                "env_scheduler_update_ms": 10.0,
                "env_step_ms": 11.0,
                "env_wall_lag_ms": 12.0,
            },
        },
    )
    return event


def test_realtime_pipeline_profile_rows_focus_on_inference_and_bench_overhead() -> None:
    row = realtime_pipeline_profile_rows([_profiled_event()])[0]

    assert row["action_id"] == 3
    assert row["obs_id"] == 7
    assert row["inference_latency_ms"] == 40.0
    assert row["env_observe_ms"] == 1.0
    assert row["parent_to_worker_ipc_ms"] == 4.0
    assert row["worker_to_parent_ipc_ms"] == 6.0
    assert row["env_step_ms"] == 11.0
    assert row["env_wall_lag_ms"] == 12.0
    assert row["bench_overhead_ms"] == 55.0
    assert "frame_quantization_delay_ms" not in row


def test_realtime_pipeline_summary_reports_stage_breakdown() -> None:
    rows = realtime_pipeline_profile_rows([_profiled_event()])
    summary = realtime_pipeline_summary(rows)

    assert summary["num_profiled_actions"] == 1
    assert summary["inference_latency_ms"]["mean"] == 40.0
    assert summary["bench_overhead_ms"]["mean"] == 55.0
    assert summary["stages"]["env_observe_ms"]["mean"] == 1.0
    assert summary["stages"]["env_step_ms"]["mean"] == 11.0


def test_trajectory_logger_writes_realtime_pipeline_profile(tmp_path) -> None:
    logger = TrajectoryLogger(
        tmp_path,
        save_step_records=False,
        save_action_records=True,
        save_latency_records=False,
    )
    logger.log_action_event(_profiled_event())

    logger.flush()

    rows = (tmp_path / "realtime_pipeline_profile.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["bench_overhead_ms"] == 55.0
    summary = json.loads((tmp_path / "realtime_pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["num_profiled_actions"] == 1
