from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from latency_bench.executors.thread_env import THREAD_ENV_DEFAULTS


def _write_config(path: Path) -> None:
    payload = {
        "experiment": {"name": "calibration-test", "seed": 0},
        "executor": {"mode": "realtime"},
        "env": {"name": "dummy", "env_fps": 16, "obs_fps": 16, "noop_action": "noop"},
        "latency": {"device": "cpu", "sync_cuda": False},
        "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
        "policy": {"type": "constant", "action": "noop", "actions": ["noop"]},
        "evaluation": {"eval_episodes": 1, "eval_parallel_envs": 1, "eval_latency_values": None, "eval_max_steps": 10},
        "logging": {"output_dir": str(path.parent / "run")},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_calibrate_env_fps_from_saturated_latency_uses_probe_samples(tmp_path, monkeypatch):
    from latency_bench.experiments import calibrate_env_fps_from_saturated_latency as module

    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "env_fps_calibration.json"
    _write_config(config_path)
    for key in THREAD_ENV_DEFAULTS:
        monkeypatch.delenv(key, raising=False)

    def fake_probe(*, config_path, samples, skip, period_ms, devices):
        assert devices == ["cuda:0", "cuda:1"]
        return {
            "period_ms": period_ms,
            "sample_count": samples,
            "skip": skip,
            "latency_samples_ms": [34.0 for _ in range(samples - skip)],
            "latency_samples_by_slot_ms": [
                [34.0 for _ in range(samples - skip)] for _ in devices
            ],
            "total_latency_ms": {"n": samples - skip, "mean": 34.0, "median": 34.0, "p90": 34.0, "p95": 34.0},
            "latency_by_slot_summary": [
                {"n": samples - skip, "mean": 34.0, "median": 34.0, "p90": 34.0, "p95": 34.0}
                for _ in devices
            ],
            "policy_build_example_ms": {"n": samples - skip, "mean": 1.0, "median": 1.0, "p90": 1.0, "p95": 1.0},
            "policy_wrapper_predict_action_ms": {"n": samples - skip, "mean": 32.0, "median": 32.0, "p90": 32.0, "p95": 32.0},
            "policy_decode_ms": {"n": samples - skip, "mean": 1.0, "median": 1.0, "p90": 1.0, "p95": 1.0},
        }

    monkeypatch.setattr(module, "_run_spawned_latency_probe", fake_probe)
    monkeypatch.setattr(module, "visible_cuda_device_count", lambda: 2)

    payload = module.calibrate_env_fps_from_saturated_latency(
        config_path=config_path,
        output=output_path,
        target_latency_frames=2,
        worker_utilization_cap=1.0,
        samples=120,
        skip=20,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert {key: os.environ[key] for key in THREAD_ENV_DEFAULTS} == THREAD_ENV_DEFAULTS
    assert written == payload
    assert payload["calibration_source"] == "saturated_policy_latency"
    assert payload["selected_env_fps"] == 58
    assert payload["saturated_latency_summary"]["median"] == 34.0
    assert payload["policy_stage_summary"]["policy_wrapper_predict_action_ms"]["median"] == 32.0
