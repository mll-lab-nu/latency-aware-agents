from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.latency.profile import load_latency_samples, load_profile_manifest
from scripts.cross_device import rebuild_profiles_from_measure_runs


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_measure_run(root: Path) -> tuple[Path, Path]:
    run_dir = (
        root
        / "runs"
        / "measure"
        / "openvla-flappy"
        / "flappy"
        / "l40s"
        / "best_seed_0_n100"
    )
    _write_jsonl(run_dir / "e2e_latencies.jsonl", [{"latency_ms": value} for value in [10.0, 20.0, 30.0, 40.0]])
    _write_jsonl(run_dir / "infer_latencies.jsonl", [{"latency_ms": value} for value in [1.0, 2.0]])
    profile = {
        "env_observe_ms": 0.0,
        "env_to_parent_ipc_ms": 0.0,
        "parent_observation_handle_ms": 0.0,
        "parent_to_worker_ipc_ms": 2.0,
        "worker_recv_overhead_ms": 3.0,
        "worker_to_parent_ipc_ms": 5.0,
        "parent_release_ms": 0.0,
        "parent_to_env_ipc_ms": 0.0,
        "env_action_recv_to_enqueue_ms": 0.0,
        "env_scheduler_update_ms": 0.0,
    }
    _write_jsonl(
        run_dir / "actions.jsonl",
        [
            {
                "action_id": 0,
                "obs_env_step": 0,
                "ready_raw_frame": 1,
                "ready_time_ms": 100.0,
                "apply_raw_frame": 1,
                "latency_raw_frames": 1,
                "applied": True,
                "dropped": False,
                "latency": {"latency_ms": 10.0},
                "metadata": {"episode_id": 0, "frame_ms": 100.0, "realtime_profile": profile},
            },
            {
                "action_id": 1,
                "obs_env_step": 1,
                "ready_raw_frame": 3,
                "ready_time_ms": 300.0,
                "apply_raw_frame": 3,
                "latency_raw_frames": 2,
                "applied": True,
                "dropped": False,
                "latency": {"latency_ms": 20.0},
                "metadata": {"episode_id": 0, "frame_ms": 100.0, "realtime_profile": profile},
            },
        ],
    )
    _write_jsonl(
        run_dir / "steps.jsonl",
        [
            {
                "episode_id": 0,
                "env_step": 0,
                "info": {"observation_submitted": True, "observation_dropped": False},
            },
            {
                "episode_id": 0,
                "env_step": 1,
                "info": {"observation_submitted": True, "observation_dropped": False},
            },
        ],
    )
    _write_jsonl(
        run_dir / "episode_metrics.jsonl",
        [
            {
                "episode_id": 0,
                "num_actions": 2,
                "metadata": {
                    "env_fps": 10.0,
                    "frame_ms": 100.0,
                    "profile_id": "L_L40S",
                    "policy_id": "starvla",
                    "env_id": "flappy",
                    "target_device": "l40s",
                    "inference_worker_count": 2,
                },
            }
        ],
    )
    profile_dir = root / "profiles" / "openvla-flappy" / "flappy" / "L_L40S"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "profile_id": "L_L40S",
                "policy_id": "openvla-flappy",
                "env_id": "flappy",
                "target_device": "l40s",
                "env_fps": 10.0,
                "frame_ms": 100.0,
                "n_samples": 1,
                "trace_path": "e2e_latencies.jsonl",
                "latency_kind": "e2e_latency",
                "distribution_path": "latency_distribution.json",
                "plot_path": "latency_distribution.png",
                "summary": {"n_samples": 1, "mean_ms": 10.0},
                "policy_type": "starvla",
                "model_type": "openvla",
                "checkpoint_path": "/tmp/model.pt",
                "hardware": {"gpu": {"name": "NVIDIA L40S"}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir, profile_path


def test_rebuild_profiles_dry_run_does_not_modify_profile(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    root = tmp_path / "results" / "cross_device"
    _run_dir, profile_path = _write_measure_run(root)
    before = profile_path.read_text(encoding="utf-8")

    rebuild_profiles_from_measure_runs.main(["--root", str(root)])

    assert profile_path.read_text(encoding="utf-8") == before
    line = capsys.readouterr().out.strip()
    payload = json.loads(line)
    assert payload["apply"] is False
    assert payload["upload"] is False
    assert payload["e2e_latency_count"] == 4
    assert payload["inference_latency_count"] == 2
    assert payload["actions_path"].endswith("actions.jsonl")
    assert payload["steps_path"].endswith("steps.jsonl")


def test_rebuild_profiles_apply_uses_full_e2e_trace_and_preserves_metadata(tmp_path: Path):
    root = tmp_path / "results" / "cross_device"
    _run_dir, profile_path = _write_measure_run(root)

    rebuild_profiles_from_measure_runs.main(["--root", str(root), "--apply"])

    profile = load_profile_manifest(profile_path)
    assert profile["n_samples"] == 4
    assert profile["e2e_summary"]["n_samples"] == 4
    assert profile["e2e_summary"]["mean_ms"] == pytest.approx(25.0)
    assert profile["inference_summary"]["n_samples"] == 2
    assert profile["inference_summary"]["mean_ms"] == pytest.approx(1.5)
    assert profile["inference_worker_count"] == 2
    assert profile["policy_id"] == "openvla-flappy"
    assert profile["policy_type"] == "starvla"
    assert profile["model_type"] == "openvla"
    assert profile["checkpoint_path"] == "/tmp/model.pt"
    assert profile["hardware"] == {"gpu": {"name": "NVIDIA L40S"}}
    assert profile["latency_method"] == "temporal"
    assert profile["sample_model_type"] == "bayes_markov"
    assert profile["burst_samples_path"] == "latency_burst_samples.jsonl"
    assert profile["burst_model_path"] == "latency_burst_model.json"
    assert profile["burst_state_key"] == "worker_service_ms_bin"
    assert load_latency_samples(profile_path.parent / "e2e_latencies.jsonl") == [10.0, 20.0, 30.0, 40.0]
    assert load_latency_samples(profile_path.parent / "infer_latencies.jsonl") == [1.0, 2.0]


def test_rebuild_profiles_uploads_profile_folder_after_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "results" / "cross_device"
    _run_dir, profile_path = _write_measure_run(root)
    calls: list[dict] = []
    monkeypatch.setattr(rebuild_profiles_from_measure_runs, "upload_folder_to_hf", lambda **kwargs: calls.append(kwargs))

    rebuild_profiles_from_measure_runs.main(
        [
            "--root",
            str(root),
            "--apply",
            "--upload",
            "--repo-id",
            "latency-sensitive-bench/profiles",
        ]
    )

    assert calls == [
        {
            "repo_id": "latency-sensitive-bench/profiles",
            "local_dir": profile_path.parent,
            "repo_type": "dataset",
            "path_in_repo": "profiles/openvla-flappy/flappy/L_L40S",
        }
    ]
