from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from latency_bench.latency.burst_profile import build_latency_burst_model


FIGURE_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "paper"
    / "latency-sensitive-vla"
    / "figures"
    / "4_1_cross_device_sim"
    / "fig_temporal_latency_model"
    / "run.py"
)


def _load_figure_module():
    spec = importlib.util.spec_from_file_location("fig_temporal_latency_model_run", FIGURE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temporal_latency_model_figure_generates_outputs(tmp_path: Path):
    profile_dir = tmp_path / "profiles" / "openvla-flappy" / "flappy" / "L_TEST"
    profile_dir.mkdir(parents=True)
    samples = [
        {"episode_id": 0, "sample_index": 0, "latency_ms": 10.0, "latency_raw_frames": 1},
        {"episode_id": 0, "sample_index": 1, "latency_ms": 11.0, "latency_raw_frames": 1},
        {"episode_id": 0, "sample_index": 2, "latency_ms": 20.0, "latency_raw_frames": 2},
        {"episode_id": 0, "sample_index": 3, "latency_ms": 21.0, "latency_raw_frames": 2},
    ]
    (profile_dir / "latency_burst_samples.jsonl").write_text(
        "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples),
        encoding="utf-8",
    )
    (profile_dir / "latency_burst_model.json").write_text(
        json.dumps(build_latency_burst_model(samples), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (profile_dir / "profile.json").write_text(
        json.dumps(
            {
                "profile_id": "L_TEST",
                "policy_id": "openvla-flappy",
                "env_id": "flappy",
                "target_device": "l40s",
                "env_fps": 25.0,
                "frame_ms": 40.0,
                "n_samples": 4,
                "trace_path": "e2e_latencies.jsonl",
                "latency_kind": "e2e_latency",
                "distribution_path": "latency_distribution.json",
                "sample_model_type": "semi_markov_latency_frames",
                "burst_samples_path": "latency_burst_samples.jsonl",
                "burst_model_path": "latency_burst_model.json",
                "burst_state_key": "latency_raw_frames",
                "summary": {},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    module = _load_figure_module()

    module.main(
        [
            "--profiles-root",
            str(tmp_path / "profiles"),
            "--output-dir",
            str(tmp_path / "figure"),
            "--devices",
            "l40s",
            "--generated-samples",
            "64",
        ]
    )

    assert (tmp_path / "figure" / "fig_temporal_latency_model.png").stat().st_size > 0
    assert (tmp_path / "figure" / "fig_temporal_latency_model.pdf").stat().st_size > 0
