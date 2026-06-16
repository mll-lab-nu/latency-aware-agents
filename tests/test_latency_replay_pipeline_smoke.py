from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from latency_bench.experiments.build_latency_profile_from_trace import build_profile_from_trace
from latency_bench.experiments.build_latency_replay_tables import main as build_latency_replay_tables_main
from latency_bench.latency.profile import write_latency_samples
from latency_bench.utils.io import write_jsonl


EXAMPLE_SPEC_PATH = Path("examples/latency_replay/table_spec.example.json")
FRAME_MS_30_FPS = 1000.0 / 30.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _build_profile(
    *,
    raw_trace: Path,
    output_dir: Path,
    profile_id: str,
    target_device: str,
) -> None:
    build_profile_from_trace(
        latencies=raw_trace,
        output_dir=output_dir,
        profile_id=profile_id,
        policy_id="flappy_pi0",
        env_id="FlappyBird-v0",
        target_device=target_device,
        env_fps=30.0,
        metadata={},
    )


def _episode_record(
    *,
    episode_return: float,
    episode_length: int,
    mode: str,
    replay_host: str | None,
    latency_type: str,
) -> dict:
    return {
        "episode_return_env": episode_return,
        "episode_return": episode_return,
        "return_raw": None,
        "survival_steps": episode_length,
        "mode": mode,
        "policy_id": "flappy_pi0",
        "env_id": "FlappyBird-v0",
        "profile_id": "L_3090",
        "target_device": "RTX_3090",
        "replay_host": replay_host,
        "env_fps": 30.0,
        "frame_ms": FRAME_MS_30_FPS,
        "latency_type": latency_type,
    }


def _write_synthetic_runs(base_dir: Path) -> tuple[Path, Path]:
    realtime_run = base_dir / "runs" / "realtime_3090"
    simulated_run = base_dir / "runs" / "sim_L3090_on_H200"
    write_jsonl(
        realtime_run / "episode_metrics.jsonl",
        [
            _episode_record(
                episode_return=100.0,
                episode_length=1000,
                mode="realtime",
                replay_host=None,
                latency_type="measured",
            ),
            _episode_record(
                episode_return=110.0,
                episode_length=1100,
                mode="realtime",
                replay_host=None,
                latency_type="measured",
            ),
        ],
    )
    write_jsonl(
        simulated_run / "episode_metrics.jsonl",
        [
            _episode_record(
                episode_return=105.0,
                episode_length=1020,
                mode="simulated",
                replay_host="H200",
                latency_type="profile_sample",
            ),
            _episode_record(
                episode_return=115.0,
                episode_length=1120,
                mode="simulated",
                replay_host="H200",
                latency_type="profile_sample",
            ),
        ],
    )
    write_jsonl(
        simulated_run / "latencies.jsonl",
        [
            {"latency_ms": 30.0},
            {"latency_ms": 40.0},
            {"latency_ms": 50.0},
        ],
    )
    return realtime_run, simulated_run


def _write_table_spec(
    *,
    spec_path: Path,
    l3090_profile: Path,
    lh200_profile: Path,
    realtime_run: Path,
    simulated_run: Path,
) -> None:
    spec = {
        "profiles": [
            str(l3090_profile / "profile.json"),
            str(lh200_profile / "profile.json"),
        ],
        "profile_matches": [
            {
                "label": "L_3090_real_vs_sim",
                "real_profile": str(l3090_profile / "profile.json"),
                "sim_latency_records": str(simulated_run / "latencies.jsonl"),
            }
        ],
        "same_device_calibration": [
            {
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
        ],
        "cross_device_prediction": [
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
        ],
    }
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_end_to_end_synthetic_latency_replay_pipeline(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    profiles_dir = tmp_path / "profiles"
    output_dir = tmp_path / "analysis_tables"
    spec_path = tmp_path / "table_spec.json"
    l3090_raw = raw_dir / "L_3090_latencies.jsonl"
    lh200_raw = raw_dir / "L_H200_latencies.jsonl"
    write_latency_samples([30.0, 40.0, 50.0, 60.0], l3090_raw)
    write_latency_samples([5.0, 8.0, 10.0, 12.0], lh200_raw)

    l3090_profile = profiles_dir / "L_3090"
    lh200_profile = profiles_dir / "L_H200"
    _build_profile(raw_trace=l3090_raw, output_dir=l3090_profile, profile_id="L_3090", target_device="RTX_3090")
    _build_profile(raw_trace=lh200_raw, output_dir=lh200_profile, profile_id="L_H200", target_device="H200")

    for profile_dir in (l3090_profile, lh200_profile):
        assert (profile_dir / "profile.json").exists()
        assert (profile_dir / "latencies.jsonl").exists()
        assert (profile_dir / "profile_summary.json").exists()
        assert (profile_dir / "profile_summary.csv").exists()

    realtime_run, simulated_run = _write_synthetic_runs(tmp_path)
    _write_table_spec(
        spec_path=spec_path,
        l3090_profile=l3090_profile,
        lh200_profile=lh200_profile,
        realtime_run=realtime_run,
        simulated_run=simulated_run,
    )

    build_latency_replay_tables_main(["--spec", str(spec_path), "--output-dir", str(output_dir)])

    expected_outputs = {
        "latency_profile_summary.csv",
        "profile_match_metrics.csv",
        "same_device_calibration.csv",
        "cross_device_prediction.csv",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_outputs

    profile_rows = _read_csv(output_dir / "latency_profile_summary.csv")
    profile_rows_by_id = {row["profile_id"]: row for row in profile_rows}
    assert set(profile_rows_by_id) == {"L_3090", "L_H200"}
    assert float(profile_rows_by_id["L_3090"]["mean_ms"]) == pytest.approx(45.0)
    assert "p95_ms" in profile_rows_by_id["L_3090"]
    assert "prob_latency_gt_1_frame" in profile_rows_by_id["L_3090"]

    calibration = _read_csv(output_dir / "same_device_calibration.csv")[0]
    assert float(calibration["realtime_mean_return"]) == pytest.approx(105.0)
    assert float(calibration["simulated_mean_return"]) == pytest.approx(110.0)
    assert float(calibration["abs_return_error"]) == pytest.approx(5.0)
    assert float(calibration["norm_return_error"]) == pytest.approx(0.025)
    assert float(calibration["realtime_mean_length"]) == pytest.approx(1050.0)
    assert float(calibration["simulated_mean_length"]) == pytest.approx(1070.0)
    assert float(calibration["abs_length_error"]) == pytest.approx(20.0)
    assert float(calibration["norm_length_error"]) == pytest.approx(0.01)

    cross_device = _read_csv(output_dir / "cross_device_prediction.csv")[0]
    assert cross_device["target_device"] == "RTX_3090"
    assert cross_device["target_profile"] == "L_3090"
    assert cross_device["replay_host"] == "H200"
    assert float(cross_device["abs_return_error"]) == pytest.approx(5.0)
    assert float(cross_device["norm_return_error"]) == pytest.approx(0.025)

    profile_match = _read_csv(output_dir / "profile_match_metrics.csv")[0]
    for key in ("w1_ms", "w1_frames", "max_tail_error"):
        assert key in profile_match
        float(profile_match[key])

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_latency_replay_tables_main(["--spec", str(spec_path), "--output-dir", str(output_dir)])

    build_latency_replay_tables_main(["--spec", str(spec_path), "--output-dir", str(output_dir), "--overwrite"])


def test_latency_replay_example_spec_json_parses():
    data = json.loads(EXAMPLE_SPEC_PATH.read_text(encoding="utf-8"))

    assert set(data) == {
        "profiles",
        "profile_matches",
        "same_device_calibration",
        "cross_device_prediction",
    }
    assert data["profiles"] == [
        "profiles/L_3090/profile.json",
        "profiles/L_H200/profile.json",
    ]
    assert "sim_latency_source" not in data["profile_matches"][0]
