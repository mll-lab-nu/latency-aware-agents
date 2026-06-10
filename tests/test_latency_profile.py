from __future__ import annotations

from pathlib import Path

import pytest

from latency_bench.latency.profile import (
    build_inverse_cdf_distribution,
    load_latency_distribution,
    load_latency_samples,
    load_profile_manifest,
    save_latency_distribution,
    save_profile_manifest,
    write_latency_samples,
)


def _profile_manifest() -> dict:
    return {
        "schema_version": 2,
        "profile_id": "L_3090_flappy_pi0",
        "profile_type": "micro_profile",
        "policy_id": "pi0",
        "env_id": "FlappyBird-v0",
        "target_device": "RTX_3090",
        "fps": 30.0,
        "frame_ms": 1000.0 / 30.0,
        "n_samples": 10000,
        "batch_size": 1,
        "trace_path": "latencies.jsonl",
        "distribution_type": "inverse_cdf",
        "distribution_path": "latency_distribution.json",
        "summary": {},
    }


def test_profile_manifest_save_and_load_preserves_fields(tmp_path: Path):
    profile_path = tmp_path / "profile.json"
    profile = _profile_manifest()

    save_profile_manifest(profile, profile_path)
    loaded = load_profile_manifest(profile_path)

    assert loaded == profile
    assert loaded["trace_path"] == "latencies.jsonl"

def test_latency_sample_write_and_load_roundtrip(tmp_path: Path):
    trace_path = tmp_path / "latencies.jsonl"

    write_latency_samples([10.0, 20.0, 30.0], trace_path)

    assert load_latency_samples(trace_path) == [10.0, 20.0, 30.0]


def test_inverse_cdf_distribution_write_and_load_roundtrip(tmp_path: Path):
    distribution_path = tmp_path / "latency_distribution.json"
    distribution = build_inverse_cdf_distribution([30.0, 10.0, 20.0])

    save_latency_distribution(distribution, distribution_path)

    loaded = load_latency_distribution(distribution_path)
    assert loaded["distribution_type"] == "inverse_cdf"
    assert loaded["quantile_levels"] == [0.0, 0.5, 1.0]
    assert loaded["latency_ms"] == [10.0, 20.0, 30.0]
def test_latency_sample_loader_rejects_empty_file(tmp_path: Path):
    trace_path = tmp_path / "empty.jsonl"
    trace_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Latency sample trace is empty"):
        load_latency_samples(trace_path)


def test_latency_sample_loader_rejects_missing_latency_key(tmp_path: Path):
    trace_path = tmp_path / "invalid.jsonl"
    trace_path.write_text('{"unexpected": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing latency_ms.*invalid.jsonl:1"):
        load_latency_samples(trace_path)
