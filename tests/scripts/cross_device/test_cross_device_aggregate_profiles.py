from __future__ import annotations

import json
from pathlib import Path

from latency_bench.latency.burst_profile import WORKER_SERVICE_STAGE_KEYS
from latency_bench.latency.profile import load_latency_samples, load_profile_manifest
from latency_bench.logging.realtime_pipeline import BENCH_OVERHEAD_STAGE_KEYS
from latency_bench.utils.io import load_jsonl_records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_existing_profile(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "profile_id": "L_3090",
                "policy_id": "openvla-flappy",
                "env_id": "flappy",
                "target_device": "rtx3090",
                "env_fps": 30.0,
                "frame_ms": 1000.0 / 30.0,
                "n_samples": 1,
                "trace_path": "e2e_latencies.jsonl",
                "latency_kind": "e2e_latency",
                "distribution_path": "latency_distribution.json",
                "plot_path": "latency_distribution.png",
                "summary": {"n_samples": 1, "mean_ms": 40.0, "frame_ms": 1000.0 / 30.0},
                "policy_type": "starvla",
                "model_type": "openvla",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_measure_run(run_dir: Path, *, instance_id: str, latencies: list[float]) -> None:
    profile = {key: 0.0 for key in set(BENCH_OVERHEAD_STAGE_KEYS) | set(WORKER_SERVICE_STAGE_KEYS)}
    _write_jsonl(run_dir / "e2e_latencies.jsonl", [{"latency_ms": value} for value in latencies])
    _write_jsonl(run_dir / "infer_latencies.jsonl", [{"latency_ms": value - 1.0} for value in latencies])
    _write_jsonl(
        run_dir / "episode_metrics.jsonl",
        [
            {
                "episode_id": index,
                "num_actions": 1,
                "metadata": {
                    "env_fps": 30.0,
                    "frame_ms": 1000.0 / 30.0,
                    "inference_worker_count": 2,
                    "profile_id": "L_3090",
                    "target_device": "rtx3090",
                    "instance_id": instance_id,
                },
            }
            for index in range(len(latencies))
        ],
    )
    _write_jsonl(
        run_dir / "actions.jsonl",
        [
            {
                "action_id": index,
                "obs_env_step": index,
                "ready_raw_frame": index + 1,
                "apply_raw_frame": index + 1,
                "latency_raw_frames": 1,
                "applied": True,
                "dropped": False,
                "latency": {"latency_ms": value},
                "metadata": {"realtime_profile": profile},
            }
            for index, value in enumerate(latencies)
        ],
    )


def test_aggregate_profiles_combines_instances_without_cross_episode_continuity(tmp_path: Path):
    from scripts.cross_device.aggregate_profiles_from_instances import aggregate_profiles

    root = tmp_path / "results"
    raw_root = root / "_hf_raw"
    profiles_dir = root / "profiles"
    profile_dir = profiles_dir / "openvla-flappy" / "flappy" / "L_3090"
    _write_existing_profile(profile_dir / "profile.json")

    first = (
        raw_root
        / "raw"
        / "rtx3090"
        / "instances"
        / "instance_a"
        / "openvla"
        / "cross_device"
        / "runs"
        / "measure"
        / "openvla-flappy"
        / "flappy"
        / "rtx3090"
        / "best_seed_0_n2"
    )
    second = (
        raw_root
        / "raw"
        / "rtx3090"
        / "instances"
        / "instance_b"
        / "openvla"
        / "cross_device"
        / "runs"
        / "measure"
        / "openvla-flappy"
        / "flappy"
        / "rtx3090"
        / "best_seed_0_n2"
    )
    _write_measure_run(first, instance_id="instance_a", latencies=[40.0, 41.0])
    _write_measure_run(second, instance_id="instance_b", latencies=[50.0, 51.0])

    report = aggregate_profiles(
        raw_root=raw_root,
        profiles_dir=profiles_dir,
        work_dir=root / "_aggregate_profiles_work",
        upload=False,
        repo_id=None,
        repo_type="dataset",
    )

    profile = load_profile_manifest(profile_dir / "profile.json")
    assert profile["n_samples"] == 4
    assert profile["env_fps"] == 30.0
    assert profile["profile_scope"] == "hardware_class_aggregate"
    assert profile["source_instance_count"] == 2
    assert profile["source_run_count"] == 2
    assert profile["source_instances"] == ["instance_a", "instance_b"]
    assert load_latency_samples(profile_dir / "e2e_latencies.jsonl") == [40.0, 41.0, 50.0, 51.0]
    burst_samples = load_jsonl_records(profile_dir / "latency_burst_samples.jsonl")
    assert [sample["episode_id"] for sample in burst_samples] == [0, 1, 2, 3]
    assert report["rebuilt_profiles"][0]["accepted_run_count"] == 2


def test_sim_script_runs_profile_aggregation_before_sim():
    text = Path("sim.sh").read_text(encoding="utf-8")

    assert 'AGGREGATE_PROFILES="${AGGREGATE_PROFILES:-1}"' in text
    assert './aggregate_profiles.sh' in text
    assert 'if [[ "$AGGREGATE_PROFILES" == "1" ]]' in text
