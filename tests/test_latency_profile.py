from __future__ import annotations

import json
from pathlib import Path

import pytest

from latency_bench.latency.profile import (
    load_latency_samples,
    load_profile_manifest,
    save_profile_manifest,
    write_latency_samples,
)


def _profile_manifest() -> dict:
    return {
        "schema_version": 1,
        "profile_id": "L_3090_flappy_pi0",
        "profile_type": "micro_profile",
        "policy_id": "pi0",
        "env_id": "FlappyBird-v0",
        "target_device": "RTX_3090",
        "fps": 30.0,
        "frame_ms": 1000.0 / 30.0,
        "n_warmup": 100,
        "n_samples": 10000,
        "batch_size": 1,
        "trace_path": "latencies.jsonl",
        "summary": {},
    }


def test_profile_manifest_save_and_load_preserves_fields(tmp_path: Path):
    profile_path = tmp_path / "profile.json"
    profile = _profile_manifest()

    save_profile_manifest(profile, profile_path)
    loaded = load_profile_manifest(profile_path)

    assert loaded == profile
    assert loaded["trace_path"] == "latencies.jsonl"

def test_profile_manifest_rejects_unknown_profile_type(tmp_path: Path):
    profile_path = tmp_path / "profile.json"
    profile = _profile_manifest()
    profile["profile_type"] = "benchmark"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported profile_type"):
        load_profile_manifest(profile_path)
def test_latency_sample_write_and_load_roundtrip(tmp_path: Path):
    trace_path = tmp_path / "latencies.jsonl"

    write_latency_samples([10.0, 20.0, 30.0], trace_path)

    assert load_latency_samples(trace_path) == [10.0, 20.0, 30.0]
def test_latency_sample_loader_rejects_empty_file(tmp_path: Path):
    trace_path = tmp_path / "empty.jsonl"
    trace_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Latency sample trace is empty"):
        load_latency_samples(trace_path)


def test_latency_sample_loader_rejects_missing_latency_key(tmp_path: Path):
    trace_path = tmp_path / "invalid.jsonl"
    trace_path.write_text('{"obs_step": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing latency_ms.*invalid.jsonl:1"):
        load_latency_samples(trace_path)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_latency_sample_writer_rejects_invalid_values(tmp_path: Path, value: float):
    trace_path = tmp_path / "latencies.jsonl"

    with pytest.raises(ValueError, match="latency_ms must be finite and non-negative"):
        write_latency_samples([value], trace_path)


@pytest.mark.parametrize(
    "line",
    [
        '{"latency_ms": -1.0}\n',
        '{"latency_ms": NaN}\n',
        '{"latency_ms": Infinity}\n',
    ],
)
def test_latency_sample_loader_rejects_invalid_values(tmp_path: Path, line: str):
    trace_path = tmp_path / "invalid_value.jsonl"
    trace_path.write_text(line, encoding="utf-8")

    with pytest.raises(ValueError, match="latency_ms must be finite and non-negative"):
        load_latency_samples(trace_path)
