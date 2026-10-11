from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
import pytest

from latency_bench.executors.thread_env import THREAD_ENV_DEFAULTS
from latency_bench.profiling.calibration import _summary


def test_calibration_summary_uses_linear_percentiles():
    summary = _summary([10.0, 20.0, 30.0, 40.0])
    assert summary["p90"] == pytest.approx(37.0)
    assert summary["p95"] == pytest.approx(38.5)


def _write_config(path: Path) -> None:
    payload = {
        'experiment': {
            'name': 'calibration-test',
            'seed': 0,
        },
        'executor': {
            'mode': 'realtime',
            'inference_devices': ['cpu', 'cuda:2'],
            'parallel_envs': 1,
            'sync_cuda': False,
        },
        'env': {
            'name': 'dummy',
            'env_fps': 16,
            'obs_fps': 16,
            'noop_action': 'noop',
        },
        'scheduler': {
            'hold_policy': 'hold',
            'ordering_policy': 'latest_ready',
        },
        'policy': {
            'type': 'constant',
            'action': 'noop',
            'actions': ['noop'],
        },
        'evaluation': {
            'episodes': 1,
            'max_env_steps': 10,
        },
        'logging': {
            'output_dir': str(path.parent / 'run'),
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_calibrate_env_fps_from_saturated_latency_uses_probe_samples(tmp_path, monkeypatch):
    from latency_bench.profiling import calibration as module

    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "env_fps_calibration.json"
    _write_config(config_path)
    for key in THREAD_ENV_DEFAULTS:
        monkeypatch.delenv(key, raising=False)

    def fake_probe(*, config_path, samples, skip, period_ms, devices):
        assert devices == ["cpu", "cuda:2"]
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
        }

    monkeypatch.setattr(module, "_run_spawned_latency_probe", fake_probe)

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
    assert payload["saturated_latency_summary"]["median"] == 34.0
