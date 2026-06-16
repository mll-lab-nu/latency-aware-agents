from __future__ import annotations

import csv
from pathlib import Path

import pytest

from latency_bench.latency.profile import write_latency_samples
from scripts.cross_device import build_figure_data as bfd


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _seed_tables(tables_dir: Path) -> None:
    _write_csv(
        tables_dir / "experiment1_latency_profile_summary.csv",
        ["experiment_id", "panel", "policy", "env", "profile_id", "hardware", "env_fps",
         "inference_mean_ms", "e2e_mean_ms", "e2e_mean_frames", "n_samples", "notes"],
        [
            {"experiment_id": "exp1", "panel": "a", "policy": "openvla-flappy", "env": "Flappy",
             "profile_id": "L_3090", "hardware": "RTX 3090", "env_fps": "30",
             "inference_mean_ms": "75.0", "e2e_mean_ms": "120.0", "e2e_mean_frames": "3.6",
             "n_samples": "70", "notes": ""},
            # Missing profile row (no mean_ms) -> dropped from table + no trace.
            {"experiment_id": "exp1", "panel": "a", "policy": "openvla-flappy", "env": "Flappy",
             "profile_id": "L_B200", "hardware": "B200", "env_fps": "",
             "inference_mean_ms": "", "e2e_mean_ms": "", "e2e_mean_frames": "",
             "n_samples": "", "notes": "missing"},
        ],
    )
    _write_csv(
        tables_dir / "experiment2_same_device_calibration.csv",
        ["experiment_id", "panel", "policy", "env", "profile_id", "hardware", "realtime_return",
         "simulated_return_using_L_d", "return_abs_error", "return_rel_error_pct", "realtime_length",
         "sim_length", "length_abs_error", "length_rel_error_pct", "n_episodes", "seed_set", "notes"],
        [
            {"experiment_id": "exp2", "panel": "b", "policy": "flappy-sf-l1", "env": "Flappy",
             "profile_id": "L_3090", "hardware": "RTX 3090", "realtime_return": "434.5",
             "simulated_return_using_L_d": "444.6", "return_abs_error": "10.0", "return_rel_error_pct": "2.3",
             "realtime_length": "3519", "sim_length": "3600", "length_abs_error": "81", "length_rel_error_pct": "2.2",
             "n_episodes": "10", "seed_set": "0", "notes": ""},
        ],
    )
    _write_csv(
        tables_dir / "experiment3_cross_device_prediction.csv",
        ["experiment_id", "panel", "policy", "env", "target_device", "target_profile", "replay_device",
         "true_realtime_return_on_d", "simulated_return_on_replay_using_L_d", "return_abs_error",
         "return_rel_error_pct", "true_realtime_length_on_d", "simulated_length_on_replay_using_L_d",
         "length_abs_error", "length_rel_error_pct", "n_episodes", "seed_set", "use_in_main_claim", "notes"],
        [
            {"experiment_id": "exp3", "panel": "c", "policy": "openvla-flappy", "env": "Flappy",
             "target_device": "RTX 3090", "target_profile": "L_3090", "replay_device": "B200",
             "true_realtime_return_on_d": "4.47", "simulated_return_on_replay_using_L_d": "3.9",
             "return_abs_error": "0.57", "return_rel_error_pct": "12.75", "true_realtime_length_on_d": "54.8",
             "simulated_length_on_replay_using_L_d": "50.0", "length_abs_error": "4.8", "length_rel_error_pct": "8.7",
             "n_episodes": "10", "seed_set": "0", "use_in_main_claim": "TRUE", "notes": ""},
        ],
    )


def test_build_all_produces_aligned_schemas(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    profiles_root = tmp_path / "profiles"
    _seed_tables(tables_dir)
    # Panel-a traces for the one device that has a profile.
    write_latency_samples(
        [float(i) for i in range(100)],
        profiles_root / "openvla-flappy" / "flappy" / "L_3090" / "e2e_latencies.jsonl",
    )
    write_latency_samples(
        [float(i + 100) for i in range(100)],
        profiles_root / "openvla-flappy" / "flappy" / "L_3090" / "infer_latencies.jsonl",
    )

    out_fig = tmp_path / "fig" / "data.csv"
    out_table = tmp_path / "table" / "data.csv"
    counts = bfd.build_all(
        tables_dir=tables_dir,
        profiles_root=profiles_root,
        out_fig=out_fig,
        out_table=out_table,
        panel_a_env="Flappy",
        max_samples_per_device=10,
    )

    fig_rows = _read_csv(out_fig)
    table_rows = _read_csv(out_table)

    # Schema matches what the plotting scripts read.
    assert list(fig_rows[0].keys()) == list(bfd.FIG_COLUMNS)
    assert list(table_rows[0].keys()) == list(bfd.TABLE3_COLUMNS)

    # Table 3: env_fps -> fps, missing-profile row dropped.
    assert len(table_rows) == 1
    assert table_rows[0]["fps"] == "30"
    assert table_rows[0]["hardware"] == "RTX 3090"
    assert table_rows[0]["inference_mean_ms"] == "75.0"
    assert table_rows[0]["e2e_mean_ms"] == "120.0"
    assert "p95_ms" not in table_rows[0]

    by_type: dict[str, list[dict[str, str]]] = {}
    for row in fig_rows:
        by_type.setdefault(row["row_type"], []).append(row)

    # Panel a: main figure samples e2e delay; inference samples are kept for the separate diagnostic figure.
    assert len(by_type["e2e_latency_sample"]) == 10
    assert len(by_type["inference_latency_sample"]) == 10
    assert {r["hardware"] for r in by_type["e2e_latency_sample"]} == {"RTX 3090"}
    assert by_type["e2e_latency_sample"][0]["panel"] == "a"
    assert by_type["e2e_latency_sample"][0]["latency_ms"] == "0.0"
    assert by_type["inference_latency_sample"][0]["latency_ms"] == "100.0"

    # Panel b: calibration row passthrough.
    assert len(by_type["calibration"]) == 1
    assert by_type["calibration"][0]["policy"] == "flappy-sf-l1"
    assert by_type["calibration"][0]["simulated_return_using_L_d"] == "444.6"

    # Panel c: target_device -> target_device_d.
    assert len(by_type["prediction"]) == 1
    assert by_type["prediction"][0]["target_device_d"] == "RTX 3090"
    assert by_type["prediction"][0]["replay_device"] == "B200"
    assert by_type["prediction"][0]["return_rel_error_pct"] == "12.75"

    assert counts == {"fig_rows": 22, "table_rows": 1}


def test_build_latency_sample_rows_requires_inference_trace_for_profiled_devices(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    profiles_root = tmp_path / "profiles"
    _seed_tables(tables_dir)
    write_latency_samples(
        [1.0, 2.0],
        profiles_root / "openvla-flappy" / "flappy" / "L_3090" / "e2e_latencies.jsonl",
    )
    exp1_rows = _read_csv(tables_dir / "experiment1_latency_profile_summary.csv")

    with pytest.raises(FileNotFoundError, match="infer_latencies.jsonl"):
        bfd.build_latency_sample_rows(
            exp1_rows,
            profiles_root,
            panel_a_env="Flappy",
            max_samples_per_device=None,
        )
