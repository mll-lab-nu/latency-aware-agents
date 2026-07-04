from __future__ import annotations

import json
from pathlib import Path

from latency_bench.latency.profile import load_profile_manifest
from scripts.cross_device import backfill_latency_burst_model


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_target(root: Path) -> tuple[Path, Path]:
    run_dir = (
        root
        / "runs"
        / "measure"
        / "openvla-flappy"
        / "flappy"
        / "l40s"
        / "best_seed_0_n100"
    )
    _write_jsonl(
        run_dir / "episode_metrics.jsonl",
        [
            {
                "episode_id": 0,
                "num_actions": 2,
                "metadata": {
                    "profile_id": "L_L40S",
                    "frame_ms": 10.0,
                },
            }
        ],
    )
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
                "ready_time_ms": 10.0,
                "apply_raw_frame": 1,
                "latency_raw_frames": 1,
                "applied": True,
                "dropped": False,
                "latency": {"latency_ms": 10.0},
                "metadata": {"frame_ms": 10.0, "realtime_profile": profile},
            },
            {
                "action_id": 1,
                "obs_env_step": 2,
                "ready_raw_frame": 5,
                "ready_time_ms": 50.0,
                "apply_raw_frame": None,
                "latency_raw_frames": 3,
                "applied": False,
                "dropped": False,
                "latency": {"latency_ms": 20.0},
                "metadata": {"frame_ms": 10.0, "realtime_profile": profile},
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
                "info": {"observation_submitted": False, "observation_dropped": True},
            },
            {
                "episode_id": 0,
                "env_step": 2,
                "info": {"observation_submitted": True, "observation_dropped": False},
            },
        ],
    )
    profile_dir = root / "profiles" / "openvla" / "flappy" / "l40s" / "instance_test"
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
                "n_samples": 2,
                "trace_path": "e2e_latencies.jsonl",
                "latency_kind": "e2e_latency",
                "distribution_path": "latency_distribution.json",
                "plot_path": "latency_distribution.png",
                "summary": {},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"profile_path": str(profile_path)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir, profile_path


def test_backfill_latency_burst_model_dry_run_does_not_modify_profile(tmp_path: Path, capsys):
    root = tmp_path / "results" / "cross_device"
    _run_dir, profile_path = _write_target(root)
    before = profile_path.read_text(encoding="utf-8")

    backfill_latency_burst_model.main(["--root", str(root)])

    assert profile_path.read_text(encoding="utf-8") == before
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["apply"] is False
    assert payload["action_count"] == 2
    assert payload["episode_count"] == 1
    assert payload["metric_action_count"] == 2


def test_backfill_latency_burst_model_apply_writes_artifacts(tmp_path: Path):
    root = tmp_path / "results" / "cross_device"
    _run_dir, profile_path = _write_target(root)

    backfill_latency_burst_model.main(["--root", str(root), "--apply"])

    profile = load_profile_manifest(profile_path)
    assert profile["latency_method"] == "temporal"
    assert profile["sample_model_type"] == "bayes_markov"
    assert profile["burst_samples_path"] == "latency_burst_samples.jsonl"
    assert profile["burst_model_path"] == "latency_burst_model.json"
    assert profile["burst_state_key"] == "worker_service_ms_bin"
    samples = [
        json.loads(line)
        for line in (profile_path.parent / "latency_burst_samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [
        {
            "action_id": sample["action_id"],
            "action_ready_ms": sample["action_ready_ms"],
            "worker_service_ms_bin": sample["worker_service_ms_bin"],
            "worker_service_ms": sample["worker_service_ms"],
            "applied": sample["applied"],
        }
        for sample in samples
    ] == [
        {
            "action_id": 0,
            "action_ready_ms": 10.0,
            "worker_service_ms_bin": 20,
            "worker_service_ms": 20.0,
            "applied": True,
        },
        {
            "action_id": 1,
            "action_ready_ms": 30.0,
            "worker_service_ms_bin": 30,
            "worker_service_ms": 30.0,
            "applied": False,
        },
    ]
    model = json.loads((profile_path.parent / "latency_burst_model.json").read_text(encoding="utf-8"))
    assert model["schema_version"] == 5
    assert model["state_key"] == "worker_service_ms_bin"
    assert model["state_counts"] == {"20": 1, "30": 1}
    assert model["model_type"] == "bayes_markov"
    assert model["emission_distributions_by_state"]["20"]["action_ready_ms"]["latency_ms"][0] == 10.0


def test_backfill_latency_burst_model_uploads_profile_artifacts(tmp_path: Path, monkeypatch):
    root = tmp_path / "results" / "cross_device"
    _run_dir, profile_path = _write_target(root)
    calls: list[dict] = []
    monkeypatch.setattr(backfill_latency_burst_model, "upload_folder_to_hf", lambda **kwargs: calls.append(kwargs))

    backfill_latency_burst_model.main(
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
            "path_in_repo": "profiles/openvla/flappy/l40s/instance_test",
            "allow_patterns": [
                "profile.json",
                "latency_burst_samples.jsonl",
                "latency_burst_model.json",
            ],
        }
    ]
