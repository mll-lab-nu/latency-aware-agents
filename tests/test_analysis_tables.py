from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from latency_bench.analysis.table_builders import (
    build_cross_device_prediction_rows,
    build_latency_profile_summary_rows,
    build_profile_match_rows,
    build_same_device_calibration_rows,
    write_rows_csv,
)
from latency_bench.experiments.build_latency_replay_tables import main
from latency_bench.latency.profile import build_inverse_cdf_distribution, save_latency_distribution, save_profile_manifest, write_latency_samples
from latency_bench.utils.io import write_jsonl


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_profile(
    base: Path,
    *,
    samples: list[float],
    profile_id: str = "L_test",
    fps: float = 25.0,
    target_device: str = "test_gpu",
) -> Path:
    profile_dir = base / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    write_latency_samples(samples, profile_dir / "latencies.jsonl")
    save_latency_distribution(
        build_inverse_cdf_distribution(samples),
        profile_dir / "latency_distribution.json",
    )
    save_profile_manifest(
        {
            "profile_id": profile_id,
            "policy_id": "pi_test",
            "env_id": "env_test",
            "target_device": target_device,
            "env_fps": fps,
            "frame_ms": 1000.0 / fps,
            "n_samples": len(samples),
            "trace_path": "latencies.jsonl",
            "distribution_path": "latency_distribution.json",
            "summary": {},
        },
        profile_dir / "profile.json",
    )
    return profile_dir


def _write_episode_metrics(path: Path, returns: list[float], lengths: list[int]) -> None:
    write_jsonl(
        path,
        [
            {
                "episode_return_env": episode_return,
                "survival_steps": length,
                "metadata": {
                    "mode": "simulated",
                    "policy_id": "flappy_pi0",
                    "env_id": "FlappyBird-v0",
                },
            }
            for episode_return, length in zip(returns, lengths)
        ],
    )


def test_build_latency_profile_summary_rows_from_profile_directory(tmp_path: Path):
    profile_dir = _write_profile(tmp_path, samples=[10.0, 20.0, 30.0], profile_id="L_summary")

    rows = build_latency_profile_summary_rows([profile_dir])

    assert len(rows) == 1
    row = rows[0]
    assert row["profile_id"] == "L_summary"
    assert row["target_device"] == "test_gpu"
    assert row["env_fps"] == pytest.approx(25.0)
    assert row["frame_ms"] == pytest.approx(40.0)
    assert row["n_samples"] == 3
    assert row["trace_path"] == "latencies.jsonl"
    assert row["mean_ms"] == pytest.approx(20.0)
    assert row["p95_frames"] == pytest.approx(29.0 / 40.0)
    assert row["prob_latency_gt_1_frame"] == pytest.approx(0.0)


def test_build_profile_match_rows_computes_distribution_metrics(tmp_path: Path):
    real_profile = _write_profile(tmp_path, samples=[10.0, 20.0, 30.0], profile_id="L_real")
    sim_latencies = tmp_path / "sim" / "latencies.jsonl"
    write_latency_samples([10.0, 20.0, 40.0], sim_latencies)

    rows = build_profile_match_rows(
        [
            {
                "label": "match_test",
                "real_profile": str(real_profile),
                "sim_latency_records": str(sim_latencies),
            }
        ]
    )

    row = rows[0]
    assert row["label"] == "match_test"
    assert row["real_profile_id"] == "L_real"
    assert row["w1_ms"] == pytest.approx(10.0 / 3.0)
    assert row["w1_frames"] == pytest.approx((10.0 / 3.0) / 40.0)
    assert row["p50_error_ms"] == pytest.approx(0.0)
    assert row["max_tail_error"] == pytest.approx(0.0)


def test_build_same_device_calibration_rows_computes_errors(tmp_path: Path):
    realtime_run = tmp_path / "real"
    simulated_run = tmp_path / "sim"
    _write_episode_metrics(realtime_run / "episode_metrics.jsonl", [100.0, 110.0], [1000, 1100])
    _write_episode_metrics(simulated_run / "episode_metrics.jsonl", [105.0, 115.0], [990, 1090])

    spec = {
        "label": "flappy_3090",
        "policy_id": "flappy_pi0",
        "env_id": "FlappyBird-v0",
        "profile_id": "L_3090",
        "hardware": "RTX_3090",
        "realtime_run": str(realtime_run),
        "simulated_run": str(simulated_run),
        "return_field": "episode_return_env",
        "length_field": "survival_steps",
        "return_scale": 200.0,
        "length_scale": 2000.0,
    }

    rows = build_same_device_calibration_rows([spec])

    row = rows[0]
    assert row["realtime_n_episodes"] == 2
    assert row["simulated_n_episodes"] == 2
    assert row["realtime_mean_return"] == pytest.approx(105.0)
    assert row["simulated_mean_return"] == pytest.approx(110.0)
    assert row["abs_return_error"] == pytest.approx(5.0)
    assert row["norm_return_error"] == pytest.approx(0.025)
    assert row["return_denominator"] == pytest.approx(200.0)
    assert row["return_denominator_mode"] == "task_scale"
    assert row["realtime_mean_length"] == pytest.approx(1050.0)
    assert row["simulated_mean_length"] == pytest.approx(1040.0)
    assert row["abs_length_error"] == pytest.approx(10.0)
    assert row["norm_length_error"] == pytest.approx(0.005)
    assert row["length_denominator_mode"] == "task_scale"

    malformed = dict(spec)
    del malformed["return_field"]
    with pytest.raises(KeyError, match="return_field"):
        build_same_device_calibration_rows([malformed])


def test_build_cross_device_prediction_rows_outputs_long_format(tmp_path: Path):
    realtime_run = tmp_path / "real"
    simulated_run = tmp_path / "sim"
    _write_episode_metrics(realtime_run / "episode_metrics.jsonl", [100.0, 110.0], [1000, 1100])
    _write_episode_metrics(simulated_run / "episode_metrics.jsonl", [105.0, 115.0], [990, 1090])

    row = build_cross_device_prediction_rows(
        [
            {
                "label": "flappy_L3090_on_H200",
                "policy_id": "flappy_pi0",
                "env_id": "FlappyBird-v0",
                "target_device": "RTX_3090",
                "target_profile": "L_3090",
                "true_realtime_run": str(realtime_run),
                "simulated_run": str(simulated_run),
                "replay_host": "H200",
                "return_field": "episode_return_env",
                "length_field": "survival_steps",
                "return_scale": 200.0,
                "length_scale": 2000.0,
            }
        ]
    )[0]

    assert row["target_device"] == "RTX_3090"
    assert row["target_profile"] == "L_3090"
    assert row["replay_host"] == "H200"
    assert row["true_realtime_n_episodes"] == 2
    assert row["simulated_n_episodes"] == 2
    assert row["true_realtime_mean_return"] == pytest.approx(105.0)
    assert row["simulated_mean_return"] == pytest.approx(110.0)
    assert row["norm_return_error"] == pytest.approx(0.025)


def test_calibration_missing_scale_behavior(tmp_path: Path):
    realtime_run = tmp_path / "real"
    simulated_run = tmp_path / "sim"
    _write_episode_metrics(realtime_run / "episode_metrics.jsonl", [100.0], [1000])
    _write_episode_metrics(simulated_run / "episode_metrics.jsonl", [110.0], [900])

    missing_scale_row = build_same_device_calibration_rows(
        [
            {
                "label": "missing_scale",
                "policy_id": "pi",
                "env_id": "env",
                "profile_id": "L",
                "hardware": "GPU",
                "realtime_run": str(realtime_run),
                "simulated_run": str(simulated_run),
                "return_field": "episode_return_env",
                "length_field": "survival_steps",
            }
        ]
    )[0]
    fallback_row = build_same_device_calibration_rows(
        [
            {
                "label": "fallback",
                "policy_id": "pi",
                "env_id": "env",
                "profile_id": "L",
                "hardware": "GPU",
                "realtime_run": str(realtime_run),
                "simulated_run": str(simulated_run),
                "return_field": "episode_return_env",
                "length_field": "survival_steps",
                "allow_realtime_denominator_fallback": True,
            }
        ]
    )[0]

    assert missing_scale_row["abs_return_error"] == pytest.approx(10.0)
    assert missing_scale_row["norm_return_error"] is None
    assert missing_scale_row["return_denominator_mode"] == "missing_scale"
    assert fallback_row["return_denominator_mode"] == "realtime_denominator"
    assert fallback_row["norm_return_error"] == pytest.approx(10.0 / (100.0 + 1e-8))


def test_write_rows_csv_rejects_empty_rows_and_uses_stable_columns(tmp_path: Path):
    path = tmp_path / "tables" / "rows.csv"

    with pytest.raises(ValueError, match="Cannot write empty CSV"):
        write_rows_csv([], path)

    write_rows_csv([{"b": 2, "a": 1}, {"a": 3, "c": 4}], path)

    assert path.read_text(encoding="utf-8").splitlines()[0] == "b,a,c"
    assert _read_csv(path) == [{"b": "2", "a": "1", "c": ""}, {"b": "", "a": "3", "c": "4"}]


def test_build_latency_replay_tables_cli_smoke_and_overwrite(tmp_path: Path):
    profile_dir = _write_profile(tmp_path / "profiles", samples=[10.0, 20.0, 30.0], profile_id="L_cli")
    realtime_run = tmp_path / "real"
    simulated_run = tmp_path / "sim"
    _write_episode_metrics(realtime_run / "episode_metrics.jsonl", [100.0, 110.0], [1000, 1100])
    _write_episode_metrics(simulated_run / "episode_metrics.jsonl", [105.0, 115.0], [990, 1090])
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "profiles": [str(profile_dir)],
                "same_device_calibration": [
                    {
                        "label": "flappy_3090",
                        "policy_id": "flappy_pi0",
                        "env_id": "FlappyBird-v0",
                        "profile_id": "L_cli",
                        "hardware": "test_gpu",
                        "realtime_run": str(realtime_run),
                        "simulated_run": str(simulated_run),
                        "return_field": "episode_return_env",
                        "length_field": "survival_steps",
                        "return_scale": 200.0,
                        "length_scale": 2000.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "tables"

    main(["--spec", str(spec_path), "--output-dir", str(out_dir)])

    assert (out_dir / "latency_profile_summary.csv").exists()
    assert (out_dir / "same_device_calibration.csv").exists()
    assert not (out_dir / "profile_match_metrics.csv").exists()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(["--spec", str(spec_path), "--output-dir", str(out_dir)])

    main(["--spec", str(spec_path), "--output-dir", str(out_dir), "--overwrite"])


def test_build_latency_replay_tables_cli_rejects_spec_with_no_sections(tmp_path: Path):
    spec_path = tmp_path / "empty.json"
    spec_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="No table sections"):
        main(["--spec", str(spec_path), "--output-dir", str(tmp_path / "tables")])
