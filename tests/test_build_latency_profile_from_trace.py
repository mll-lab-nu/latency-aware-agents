from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from latency_bench.experiments.build_latency_profile_from_trace import build_profile_from_trace
from latency_bench.latency.profile import load_latency_samples, load_profile_manifest, write_latency_samples


def _build_profile(tmp_path: Path, samples: list[float], **overrides):
    input_trace = tmp_path / "input_latencies.jsonl"
    output_dir = tmp_path / "profile"
    write_latency_samples(samples, input_trace)

    kwargs = {
        "latencies": input_trace,
        "output_dir": output_dir,
        "profile_id": "L_test",
        "profile_type": "rollout_profile",
        "policy_id": "pi_test",
        "env_id": "env_test",
        "target_device": "test_gpu",
        "fps": 25.0,
        "batch_size": 1,
        "overwrite": False,
    }
    kwargs.update(overrides)
    return build_profile_from_trace(**kwargs), output_dir


def test_build_latency_profile_from_trace_writes_profile_artifacts(tmp_path: Path):
    manifest, output_dir = _build_profile(tmp_path, [10.0, 20.0, 30.0])

    assert (output_dir / "profile.json").exists()
    assert (output_dir / "latencies.jsonl").exists()
    assert (output_dir / "profile_summary.json").exists()
    assert (output_dir / "profile_summary.csv").exists()

    loaded_manifest = load_profile_manifest(output_dir / "profile.json")
    assert manifest == loaded_manifest
    assert loaded_manifest["n_samples"] == 3
    assert loaded_manifest["frame_ms"] == pytest.approx(40.0)
    assert loaded_manifest["trace_path"] == "latencies.jsonl"
    assert loaded_manifest["summary"]["mean_ms"] == pytest.approx(20.0)
    assert load_latency_samples(output_dir / "latencies.jsonl") == [10.0, 20.0, 30.0]

    summary_json = json.loads((output_dir / "profile_summary.json").read_text(encoding="utf-8"))
    assert summary_json["n_samples"] == 3
    assert summary_json["mean_ms"] == pytest.approx(20.0)

    with (output_dir / "profile_summary.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]["mean_ms"]) == pytest.approx(20.0)


def test_build_latency_profile_from_trace_requires_overwrite_for_existing_artifacts(tmp_path: Path):
    _, output_dir = _build_profile(tmp_path, [10.0, 20.0, 30.0])

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing profile artifacts"):
        _build_profile(tmp_path, [10.0, 20.0, 30.0], output_dir=output_dir)

    manifest, _ = _build_profile(tmp_path, [30.0, 40.0], output_dir=output_dir, overwrite=True)

    assert manifest["n_samples"] == 2
    assert load_latency_samples(output_dir / "latencies.jsonl") == [30.0, 40.0]


def test_build_latency_profile_from_trace_rejects_invalid_fps(tmp_path: Path):
    input_trace = tmp_path / "input_latencies.jsonl"
    write_latency_samples([10.0], input_trace)

    with pytest.raises(ValueError, match="fps must be positive"):
        build_profile_from_trace(
            latencies=input_trace,
            output_dir=tmp_path / "profile",
            profile_id="L_test",
            profile_type="rollout_profile",
            policy_id="pi_test",
            env_id="env_test",
            target_device="test_gpu",
            fps=0.0,
        )
