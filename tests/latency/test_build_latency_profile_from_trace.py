from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from latency_bench.experiments.build_latency_profile_from_trace import build_profile_from_trace
from latency_bench.latency.profile import load_latency_distribution, load_latency_samples, load_profile_manifest, write_latency_samples
from latency_bench.utils.io import load_jsonl_records


def _build_profile(tmp_path: Path, samples: list[float], **overrides):
    latency_kind = overrides.pop("latency_kind", "inference_latency")
    input_trace = tmp_path / "input_latencies.jsonl"
    output_dir = tmp_path / "profile"
    write_latency_samples(samples, input_trace)

    kwargs = {
        "latencies": input_trace,
        "output_dir": output_dir,
        "profile_id": "L_test",
        "policy_id": "pi_test",
        "env_id": "env_test",
        "target_device": "test_gpu",
        "env_fps": 25.0,
        "metadata": {
            "policy_type": "starvla",
            "model_type": "openvla",
            "checkpoint_path": "/tmp/openvla.pt",
        },
        "latency_kind": latency_kind,
        "overwrite": False,
    }
    kwargs.update(overrides)
    return build_profile_from_trace(**kwargs), output_dir


_ZERO_IPC_PROFILE = {
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


def _write_actions(path: Path) -> None:
    rows = [
        {
            "action_id": 0,
            "obs_env_step": 0,
            "ready_raw_frame": 1,
            "ready_time_ms": 40.0,
            "apply_raw_frame": 1,
            "latency_raw_frames": 1,
            "applied": True,
            "dropped": False,
            "latency": {"latency_ms": 10.0},
            "metadata": {"frame_ms": 40.0, "realtime_profile": _ZERO_IPC_PROFILE},
        },
        {
            "action_id": 1,
            "obs_env_step": 1,
            "ready_raw_frame": 2,
            "ready_time_ms": 80.0,
            "apply_raw_frame": 2,
            "latency_raw_frames": 1,
            "applied": True,
            "dropped": False,
            "latency": {"latency_ms": 11.0},
            "metadata": {"frame_ms": 40.0, "realtime_profile": _ZERO_IPC_PROFILE},
        },
        {
            "action_id": 2,
            "obs_env_step": 2,
            "ready_raw_frame": 4,
            "ready_time_ms": 160.0,
            "apply_raw_frame": 4,
            "latency_raw_frames": 2,
            "applied": True,
            "dropped": False,
            "latency": {"latency_ms": 50.0},
            "metadata": {"frame_ms": 40.0, "realtime_profile": _ZERO_IPC_PROFILE},
        },
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_episode_metrics(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")


def test_build_latency_profile_from_trace_writes_profile_artifacts(tmp_path: Path):
    manifest, output_dir = _build_profile(
        tmp_path,
        [10.0, 20.0, 30.0],
    )

    assert (output_dir / "profile.json").exists()
    assert (output_dir / "infer_latencies.jsonl").exists()
    assert (output_dir / "latency_distribution.json").exists()
    assert (output_dir / "latency_distribution.png").stat().st_size > 0
    assert (output_dir / "profile_summary.json").exists()
    assert (output_dir / "profile_summary.csv").exists()

    loaded_manifest = load_profile_manifest(output_dir / "profile.json")
    assert manifest == loaded_manifest
    assert set(loaded_manifest) == {
        "profile_id",
        "policy_id",
        "env_id",
        "target_device",
        "env_fps",
        "policy_type",
        "model_type",
        "checkpoint_path",
        "frame_ms",
        "n_samples",
        "trace_path",
        "latency_kind",
        "distribution_path",
        "plot_path",
        "summary",
    }
    assert loaded_manifest["n_samples"] == 3
    assert loaded_manifest["env_fps"] == pytest.approx(25.0)
    assert loaded_manifest["frame_ms"] == pytest.approx(40.0)
    assert loaded_manifest["trace_path"] == "infer_latencies.jsonl"
    assert loaded_manifest["latency_kind"] == "inference_latency"
    assert loaded_manifest["distribution_path"] == "latency_distribution.json"
    assert loaded_manifest["plot_path"] == "latency_distribution.png"
    assert loaded_manifest["policy_type"] == "starvla"
    assert loaded_manifest["model_type"] == "openvla"
    assert loaded_manifest["checkpoint_path"] == "/tmp/openvla.pt"
    assert loaded_manifest["summary"]["mean_ms"] == pytest.approx(20.0)
    assert load_latency_samples(output_dir / "infer_latencies.jsonl") == [10.0, 20.0, 30.0]
    distribution = load_latency_distribution(output_dir / "latency_distribution.json")
    assert distribution["quantile_levels"] == [0.0, 0.5, 1.0]
    assert distribution["latency_ms"] == [10.0, 20.0, 30.0]

    summary_json = json.loads((output_dir / "profile_summary.json").read_text(encoding="utf-8"))
    assert summary_json["n_samples"] == 3
    assert summary_json["mean_ms"] == pytest.approx(20.0)

    with (output_dir / "profile_summary.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]["mean_ms"]) == pytest.approx(20.0)


def test_build_latency_profile_from_trace_writes_latency_burst_model(tmp_path: Path):
    actions = tmp_path / "actions.jsonl"
    episode_metrics = tmp_path / "episode_metrics.jsonl"
    _write_actions(actions)
    _write_episode_metrics(
        episode_metrics,
        [
            {
                "episode_id": 0,
                "num_actions": 3,
                "metadata": {
                    "inference_worker_count": 2,
                    "frame_ms": 40.0,
                },
            }
        ],
    )

    manifest, output_dir = _build_profile(
        tmp_path,
        [10.0, 11.0, 20.0],
        actions=actions,
        episode_metrics=episode_metrics,
    )

    assert (output_dir / "latency_burst_samples.jsonl").exists()
    assert (output_dir / "latency_burst_model.json").exists()
    assert manifest["sample_model_type"] == "hidden_regime"
    assert manifest["burst_samples_path"] == "latency_burst_samples.jsonl"
    assert manifest["burst_model_path"] == "latency_burst_model.json"
    burst_samples = load_jsonl_records(output_dir / "latency_burst_samples.jsonl")
    assert [sample["action_ready_ms"] for sample in burst_samples] == [40.0, 40.0, 80.0]
    assert [sample["worker_service_ms_bin"] for sample in burst_samples] == [10, 11, 50]
    assert [sample["worker_service_ms"] for sample in burst_samples] == [10.0, 11.0, 50.0]
    burst_model = json.loads((output_dir / "latency_burst_model.json").read_text(encoding="utf-8"))
    assert burst_model["schema_version"] == 6
    assert burst_model["model_type"] == "hidden_regime"
    # Median worker-service is 11ms -> spike threshold 13.75ms -> the 50ms
    # record is a length-1 burst.
    assert burst_model["spike_threshold_ms"] == pytest.approx(13.75)
    assert burst_model["regime_step_counts"] == {"calm": 2, "burst": 1}
    assert burst_model["regime_transition_counts"] == {
        "calm": {"calm": 1, "burst": 1},
        "burst": {"calm": 0, "burst": 0},
    }
    assert burst_model["burst_dwell_lengths"] == [1]
    assert manifest["temporal_summary"]["burst_regime"]["burst_count"] == 1


def test_build_latency_profile_from_trace_requires_overwrite_for_existing_artifacts(tmp_path: Path):
    _, output_dir = _build_profile(tmp_path, [10.0, 20.0, 30.0])

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing profile artifacts"):
        _build_profile(tmp_path, [10.0, 20.0, 30.0], output_dir=output_dir)

    manifest, _ = _build_profile(tmp_path, [30.0, 40.0], output_dir=output_dir, overwrite=True)

    assert manifest["n_samples"] == 2
    assert load_latency_samples(output_dir / "infer_latencies.jsonl") == [30.0, 40.0]


def test_build_latency_profile_from_trace_plots_constant_latency(tmp_path: Path):
    manifest, output_dir = _build_profile(tmp_path, [42.0, 42.0, 42.0])

    assert manifest["n_samples"] == 3
    assert (output_dir / "latency_distribution.png").stat().st_size > 0


def test_build_latency_profile_from_trace_uses_all_samples(tmp_path: Path):
    manifest, output_dir = _build_profile(tmp_path, [10.0, 20.0, 30.0, 40.0])

    assert manifest["n_samples"] == 4
    assert load_latency_samples(output_dir / "infer_latencies.jsonl") == [10.0, 20.0, 30.0, 40.0]
    distribution = load_latency_distribution(output_dir / "latency_distribution.json")
    assert distribution["n_samples"] == 4
    assert distribution["latency_ms"] == [10.0, 20.0, 30.0, 40.0]
    summary_json = json.loads((output_dir / "profile_summary.json").read_text(encoding="utf-8"))
    assert summary_json["n_samples"] == 4
    assert summary_json["mean_ms"] == pytest.approx(25.0)


def test_build_latency_profile_from_trace_rejects_invalid_fps(tmp_path: Path):
    input_trace = tmp_path / "input_latencies.jsonl"
    write_latency_samples([10.0], input_trace)

    with pytest.raises(ZeroDivisionError):
        build_profile_from_trace(
            latencies=input_trace,
            output_dir=tmp_path / "profile",
            profile_id="L_test",
            policy_id="pi_test",
            env_id="env_test",
            target_device="test_gpu",
            env_fps=0.0,
            metadata={},
            latency_kind="inference_latency",
        )


def test_build_latency_profile_from_trace_writes_e2e_trace_name(tmp_path: Path):
    raw_inference = tmp_path / "raw_infer_latencies.jsonl"
    write_latency_samples([10.0, 20.0], raw_inference)
    manifest, output_dir = _build_profile(
        tmp_path,
        [40.0, 80.0],
        latency_kind="e2e_latency",
        inference_latencies=raw_inference,
    )

    assert (output_dir / "e2e_latencies.jsonl").exists()
    assert (output_dir / "infer_latencies.jsonl").exists()
    assert manifest["trace_path"] == "e2e_latencies.jsonl"
    assert manifest["latency_kind"] == "e2e_latency"
    assert manifest["inference_trace_path"] == "infer_latencies.jsonl"
    assert manifest["summary"]["mean_ms"] == pytest.approx(60.0)
    assert manifest["inference_summary"]["mean_ms"] == pytest.approx(15.0)
    assert load_latency_samples(output_dir / "e2e_latencies.jsonl") == [40.0, 80.0]
    assert load_latency_samples(output_dir / "infer_latencies.jsonl") == [10.0, 20.0]


def test_build_latency_profile_from_trace_copies_inference_worker_count_from_episode_metrics(tmp_path: Path):
    episode_metrics = tmp_path / "episode_metrics.jsonl"
    episode_metrics.write_text(
        json.dumps(
            {
                "episode_id": 0,
                "metadata": {
                    "inference_worker_count": 2,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest, output_dir = _build_profile(
        tmp_path,
        [10.0, 20.0, 30.0],
        episode_metrics=episode_metrics,
    )

    assert manifest["inference_worker_count"] == 2
    loaded_manifest = load_profile_manifest(output_dir / "profile.json")
    assert loaded_manifest["inference_worker_count"] == 2


def test_build_latency_profile_from_trace_summarizes_control_and_worker_capacity(tmp_path: Path):
    actions = tmp_path / "actions.jsonl"
    steps = tmp_path / "steps.jsonl"
    episode_metrics = tmp_path / "episode_metrics.jsonl"
    _write_actions(actions)
    _write_episode_metrics(
        episode_metrics,
        [
            {
                "episode_id": 0,
                "episode_return": 10.0,
                "survival_steps": 100,
                "num_actions": 2,
                "num_dropped_actions": 1,
                "metadata": {
                    "inference_worker_count": 2,
                    "submitted_observation_frames": 100,
                    "dropped_observation_count": 5,
                    "ignored_late_result_count": 1,
                    "discarded_ready_result_count": 2,
                    "fifo_blocked_count": 3,
                    "actual_env_fps": 29.0,
                    "env_deadline_miss_count": 4,
                    "max_wall_clock_lag_ms": 12.5,
                    "frame_ms": 40.0,
                },
            },
            {
                "episode_id": 1,
                "episode_return": 20.0,
                "survival_steps": 200,
                "num_actions": 1,
                "num_dropped_actions": 0,
                "metadata": {
                    "inference_worker_count": 2,
                    "submitted_observation_frames": 200,
                    "dropped_observation_count": 1,
                    "ignored_late_result_count": 0,
                    "discarded_ready_result_count": 1,
                    "fifo_blocked_count": 0,
                    "actual_env_fps": 30.0,
                    "env_deadline_miss_count": 0,
                    "max_wall_clock_lag_ms": 2.5,
                    "frame_ms": 40.0,
                },
            },
        ],
    )
    steps.write_text(
        "\n".join(
            json.dumps({"env_step": idx, "info": {"wall_clock_lag_ms": float(idx)}})
            for idx in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "realtime_pipeline_summary.json").write_text(
        json.dumps({"num_profiled_actions": 3, "bench_overhead_ms": {"mean": 1.5}}, sort_keys=True),
        encoding="utf-8",
    )
    (tmp_path / "runtime_telemetry_summary.json").write_text(
        json.dumps({"num_samples": 2, "gpu": {"utilization.gpu": {"mean": 80.0}}}, sort_keys=True),
        encoding="utf-8",
    )

    manifest, output_dir = _build_profile(
        tmp_path,
        [40.0, 50.0, 60.0],
        actions=actions,
        episode_metrics=episode_metrics,
        steps=steps,
    )

    assert manifest["control_summary"]["episodes"] == 2
    assert manifest["control_summary"]["mean_return"] == pytest.approx(15.0)
    assert manifest["control_summary"]["mean_survival_steps"] == pytest.approx(150.0)
    assert manifest["control_summary"]["submitted_observation_frames"] == 300
    assert manifest["control_summary"]["dropped_observation_count"] == 6
    assert manifest["control_summary"]["drop_rate_per_1k_observations"] == pytest.approx(20.0)
    assert manifest["control_summary"]["actions_per_step"] == pytest.approx(3.0 / 300.0)
    assert manifest["control_summary"]["ignored_late_result_count"] == 1
    assert manifest["control_summary"]["discarded_ready_result_count"] == 3
    assert manifest["control_summary"]["fifo_blocked_count"] == 3
    assert manifest["control_summary"]["env_deadline_miss_count"] == 4
    assert manifest["control_summary"]["max_wall_clock_lag_ms"] == pytest.approx(12.5)
    assert manifest["worker_capacity_summary"]["inference_worker_count"] == 2
    assert manifest["worker_capacity_summary"]["mean_worker_service_ms"] == pytest.approx((10.0 + 11.0 + 50.0) / 3.0)
    assert manifest["worker_capacity_summary"]["worker_utilization_estimate"] == pytest.approx(
        ((10.0 + 11.0 + 50.0) / 3.0) * 25.0 / 1000.0 / 2.0
    )
    assert manifest["pipeline_summary_path"] == "realtime_pipeline_summary.json"
    assert manifest["runtime_telemetry_summary_path"] == "runtime_telemetry_summary.json"
    assert manifest["raw_artifacts"]["actions"] == "actions.jsonl"
    assert manifest["raw_artifacts"]["steps"] == "steps.jsonl"
    assert manifest["raw_artifacts"]["episode_metrics"] == "episode_metrics.jsonl"
    loaded_manifest = load_profile_manifest(output_dir / "profile.json")
    assert loaded_manifest["control_summary"] == manifest["control_summary"]


def test_build_latency_profile_from_trace_requires_inference_trace_for_e2e_profiles(tmp_path: Path):
    with pytest.raises(TypeError):
        _build_profile(
            tmp_path,
            [40.0, 80.0],
            latency_kind="e2e_latency",
        )
