from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from latency_bench.experiments.build_latency_profile_from_trace import build_profile_from_trace
from latency_bench.latency.profile import load_latency_distribution, load_latency_samples, load_profile_manifest, write_latency_samples


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
    assert manifest["e2e_trace_path"] == "e2e_latencies.jsonl"
    assert manifest["inference_trace_path"] == "infer_latencies.jsonl"
    assert manifest["e2e_summary"]["mean_ms"] == pytest.approx(60.0)
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


def test_build_latency_profile_from_trace_requires_inference_trace_for_e2e_profiles(tmp_path: Path):
    with pytest.raises(TypeError):
        _build_profile(
            tmp_path,
            [40.0, 80.0],
            latency_kind="e2e_latency",
        )
