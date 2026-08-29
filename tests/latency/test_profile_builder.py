import json
from pathlib import Path

import pytest

from latency_bench.latency.profile_builder import build_profile_from_traces


def _write_session(
    root: Path,
    session_id: int,
    service_ms: list[float],
    *,
    dropped_indices: tuple[int, ...] = (),
) -> dict:
    session = root / str(session_id)
    session.mkdir(parents=True)
    actions = session / "actions.jsonl"
    metrics = session / "episode_metrics.jsonl"
    def pipeline_profile(index: int) -> dict:
        profile = {
            "env_observe_ms": 0.0,
            "env_to_parent_ipc_ms": 0.0,
            "parent_observation_handle_ms": 0.0,
            "parent_to_worker_ipc_ms": 0.0,
            "worker_recv_overhead_ms": 0.0,
            "worker_to_parent_ipc_ms": 0.0,
            "parent_release_ms": 0.0,
            "parent_to_env_ipc_ms": 0.0,
            "env_action_recv_to_enqueue_ms": 0.0,
            "env_scheduler_update_ms": 0.0,
        }
        if index in dropped_indices:
            del profile["parent_to_env_ipc_ms"]
            del profile["env_action_recv_to_enqueue_ms"]
            del profile["env_scheduler_update_ms"]
        return profile

    actions.write_text(
        "".join(
            json.dumps(
                {
                    "action_id": index,
                    "obs_env_step": index,
                    "ready_raw_frame": index + 1,
                    "ready_time_ms": index * 40.0 + value,
                    "applied": index not in dropped_indices,
                    "dropped": index in dropped_indices,
                    "latency": {
                        "latency_ms": value,
                        "metadata": {"worker_id": index % 2},
                    },
                    "metadata": {
                        "frame_ms": 40.0,
                        "realtime_profile": pipeline_profile(index),
                    },
                }
            )
            + "\n"
            for index, value in enumerate(service_ms)
        ),
        encoding="utf-8",
    )
    metrics.write_text(
        json.dumps(
            {"episode_id": 0, "num_actions": len(service_ms)}
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "measure_session_id": session_id,
        "episode_metrics": metrics,
        "actions": actions,
    }


def test_profile_builder_emits_per_slot_bundle_and_preserves_session_boundaries(
    tmp_path: Path,
):
    sessions = [
        _write_session(
            tmp_path / "sessions",
            0,
            [10.0, 11.0, 50.0, 12.0],
            dropped_indices=(2,),
        ),
        _write_session(tmp_path / "sessions", 1, [10.0, 11.0, 60.0, 12.0]),
    ]
    output = tmp_path / "profile"

    profile = build_profile_from_traces(
        sessions=sessions,
        output_dir=output,
        model_id="openvla",
        gpu_class="2x-rtx4090",
        workload_id="flappy",
        instance_id="instance_abc",
        source_run_id="run_1",
        env_fps=25.0,
        metadata={"session_ids": [0, 1]},
    )

    assert {path.name for path in output.iterdir()} == {
        "profile.json",
        "latency_distribution.json",
        "latency_burst_model.json",
    }
    assert profile["source_run_id"] == "run_1"
    assert profile["n_samples"] == 8
    assert profile["summary"]["n_samples"] == 7
    assert profile["n_action_ready_samples"] == 7
    assert profile["n_dropped_samples"] == 1
    assert set(profile["per_slot_summary"]) == {"0", "1"}
    assert (
        profile["per_slot_summary"]["0"]["mean_worker_service_ms"]
        > profile["per_slot_summary"]["1"]["mean_worker_service_ms"]
    )
    assert profile["per_slot_summary"]["0"]["action_ready_count"] == 3
    assert profile["per_slot_summary"]["0"]["dropped_count"] == 1
    burst = json.loads((output / "latency_burst_model.json").read_text())
    assert burst["burst_dwell_lengths"] == [1, 1]
    assert len(burst["burst_slot_rank_templates"]) == 2

    with pytest.raises(FileExistsError):
        build_profile_from_traces(
            sessions=sessions,
            output_dir=output,
            model_id="openvla",
            gpu_class="2x-rtx4090",
            workload_id="flappy",
            instance_id="instance_abc",
            source_run_id="run_1",
            env_fps=25.0,
            metadata={"session_ids": [0, 1]},
        )
