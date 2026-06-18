from __future__ import annotations

import csv
from pathlib import Path

import pytest

from latency_bench.latency.profile import write_latency_samples
from latency_bench.utils.io import read_csv, write_jsonl
from scripts.cross_device import build_figure_data as bfd


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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


def _spec() -> dict:
    return {
        "devices": [
            {
                "id": "rtx3090",
                "display_name": "RTX 3090",
                "profile_id": "L_3090",
            },
        ],
        "experiments": {
            "exp2_same_device_calibration": {
                "rows": [
                    {
                        "policy": "flappy-sf-l1",
                        "env": "Flappy",
                    },
                ],
            },
        },
    }


def _manifest(output_dir: Path, *, stage: str = "measure") -> dict:
    return {
        "stage": stage,
        "status": "completed",
        "experiment_id": "exp1_latency_profile_summary",
        "experiment_ids": [
            "exp1_latency_profile_summary",
            "exp2_same_device_calibration",
            "exp3_cross_device_prediction",
        ],
        "policy": "flappy-sf-l1",
        "env": "Flappy",
        "target_device_id": "rtx3090",
        "output_dir": str(output_dir),
        "n_episodes": 2,
        "seed_set": [0],
        "notes": "",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
    }


def _write_episode_metrics(
    output_dir: Path,
    *,
    returns: tuple[float, float] = (10.0, 20.0),
    lengths: tuple[int, int] = (100, 200),
) -> None:
    write_jsonl(
        output_dir / "episode_metrics.jsonl",
        [
            {"episode_id": 0, "episode_return_env": returns[0], "survival_steps": lengths[0]},
            {"episode_id": 1, "episode_return_env": returns[1], "survival_steps": lengths[1]},
        ],
    )


def test_build_all_produces_aligned_schemas(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    profiles_root = tmp_path / "profiles"
    root = tmp_path / "cross_device"
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
    run_dir = root / "runs" / "measure" / "flappy-sf-l1" / "flappy" / "rtx3090" / "best_seed_0_n2"
    _write_episode_metrics(run_dir)
    sim_dir = root / "runs" / "sim_same" / "flappy-sf-l1" / "flappy" / "rtx3090" / "best_seed_0_n2"
    _write_episode_metrics(sim_dir, returns=(30.0, 40.0), lengths=(300, 400))
    write_jsonl(root / "manifests" / "runs.jsonl", [_manifest(run_dir), _manifest(sim_dir, stage="sim_same")])

    out_fig = tmp_path / "fig" / "data.csv"
    out_table = tmp_path / "table" / "data.csv"
    counts = bfd.build_all(
        tables_dir=tables_dir,
        profiles_root=profiles_root,
        out_fig=out_fig,
        out_table=out_table,
        panel_a_env="Flappy",
        spec=_spec(),
        root=root,
    )

    fig_rows = read_csv(out_fig)
    table_rows = read_csv(out_table)

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
    assert len(by_type["e2e_latency_sample"]) == 100
    assert len(by_type["inference_latency_sample"]) == 100
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

    # Return distribution: one row per realtime measure episode.
    assert len(by_type["realtime_episode_return"]) == 2
    assert [row["episode_id"] for row in by_type["realtime_episode_return"]] == ["0", "1"]
    assert [row["episode_return_env"] for row in by_type["realtime_episode_return"]] == ["10.0", "20.0"]
    assert [row["survival_steps"] for row in by_type["realtime_episode_return"]] == ["100", "200"]
    assert {row["target_device_id"] for row in by_type["realtime_episode_return"]} == {"rtx3090"}
    assert {row["hardware"] for row in by_type["realtime_episode_return"]} == {"RTX 3090"}

    # Return distribution: one row per same-device simulated replay episode.
    assert len(by_type["simulated_episode_return"]) == 2
    assert [row["episode_id"] for row in by_type["simulated_episode_return"]] == ["0", "1"]
    assert [row["episode_return_env"] for row in by_type["simulated_episode_return"]] == ["30.0", "40.0"]
    assert [row["survival_steps"] for row in by_type["simulated_episode_return"]] == ["300", "400"]
    assert [row["simulated_return_using_L_d"] for row in by_type["simulated_episode_return"]] == ["30.0", "40.0"]
    assert [row["sim_length"] for row in by_type["simulated_episode_return"]] == ["300", "400"]
    assert {row["target_device_id"] for row in by_type["simulated_episode_return"]} == {"rtx3090"}
    assert {row["hardware"] for row in by_type["simulated_episode_return"]} == {"RTX 3090"}

    assert counts == {"fig_rows": 206, "table_rows": 1}


def test_realtime_episode_return_rows_read_hf_raw_layout(tmp_path: Path):
    raw_root = tmp_path / "raw"
    protocol_root = raw_root / "rtx3090" / "openvla" / "cross_device"
    old_root = Path("results/cross_device")
    run_dir = protocol_root / "runs" / "measure" / "flappy-sf-l1" / "flappy" / "rtx3090" / "best_seed_0_n2"
    _write_episode_metrics(run_dir)
    manifest = _manifest(old_root / run_dir.relative_to(protocol_root))
    write_jsonl(protocol_root / "manifests" / "runs.jsonl", [manifest])

    rows = bfd.build_realtime_episode_return_rows(
        _spec(),
        root=tmp_path / "unused_cross_device",
        raw_root=raw_root,
    )

    assert len(rows) == 2
    assert [row["episode_id"] for row in rows] == [0, 1]
    assert [row["episode_return_env"] for row in rows] == [10.0, 20.0]
    assert [row["survival_steps"] for row in rows] == [100, 200]
    assert {row["hardware"] for row in rows} == {"RTX 3090"}


def test_simulated_episode_return_rows_read_hf_raw_layout(tmp_path: Path):
    raw_root = tmp_path / "raw"
    protocol_root = raw_root / "rtx3090" / "openvla" / "cross_device"
    old_root = Path("results/cross_device")
    run_dir = protocol_root / "runs" / "sim_same" / "flappy-sf-l1" / "flappy" / "rtx3090" / "best_seed_0_n2"
    _write_episode_metrics(run_dir, returns=(30.0, 40.0), lengths=(300, 400))
    manifest = _manifest(old_root / run_dir.relative_to(protocol_root), stage="sim_same")
    write_jsonl(protocol_root / "manifests" / "runs.jsonl", [manifest])

    rows = bfd.build_simulated_episode_return_rows(
        _spec(),
        root=tmp_path / "unused_cross_device",
        raw_root=raw_root,
    )

    assert len(rows) == 2
    assert [row["episode_id"] for row in rows] == [0, 1]
    assert [row["episode_return_env"] for row in rows] == [30.0, 40.0]
    assert [row["survival_steps"] for row in rows] == [300, 400]
    assert [row["simulated_return_using_L_d"] for row in rows] == [30.0, 40.0]
    assert [row["sim_length"] for row in rows] == [300, 400]
    assert {row["hardware"] for row in rows} == {"RTX 3090"}


def test_profiled_e2e_trace_emits_overhead_samples(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    profiles_root = tmp_path / "profiles"
    _seed_tables(tables_dir)
    # Profiled measure run: e2e records carry bench_overhead_ms alongside latency_ms.
    write_jsonl(
        profiles_root / "openvla-flappy" / "flappy" / "L_3090" / "e2e_latencies.jsonl",
        [{"latency_ms": float(i), "bench_overhead_ms": float(i) + 5.0} for i in range(100)],
    )
    write_latency_samples(
        [float(i + 100) for i in range(100)],
        profiles_root / "openvla-flappy" / "flappy" / "L_3090" / "infer_latencies.jsonl",
    )
    exp1_rows = read_csv(tables_dir / "experiment1_latency_profile_summary.csv")

    rows = bfd.build_latency_sample_rows(
        exp1_rows,
        profiles_root,
        panel_a_env="Flappy",
    )
    overhead = [row for row in rows if row["row_type"] == "overhead_latency_sample"]
    assert len(overhead) == 100
    assert {row["hardware"] for row in overhead} == {"RTX 3090"}
    assert overhead[0]["panel"] == "a"
    assert overhead[0]["latency_ms"] == 5.0  # bench_overhead_ms of sample 0


def test_unprofiled_e2e_trace_emits_no_overhead_samples(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    profiles_root = tmp_path / "profiles"
    _seed_tables(tables_dir)
    # Legacy quantized trace: latency_ms only, no bench_overhead_ms.
    write_latency_samples(
        [float(i) for i in range(100)],
        profiles_root / "openvla-flappy" / "flappy" / "L_3090" / "e2e_latencies.jsonl",
    )
    write_latency_samples(
        [float(i + 100) for i in range(100)],
        profiles_root / "openvla-flappy" / "flappy" / "L_3090" / "infer_latencies.jsonl",
    )
    exp1_rows = read_csv(tables_dir / "experiment1_latency_profile_summary.csv")

    rows = bfd.build_latency_sample_rows(
        exp1_rows,
        profiles_root,
        panel_a_env="Flappy",
    )
    assert not [row for row in rows if row["row_type"] == "overhead_latency_sample"]


def test_build_latency_sample_rows_requires_inference_trace_for_profiled_devices(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    profiles_root = tmp_path / "profiles"
    _seed_tables(tables_dir)
    write_latency_samples(
        [1.0, 2.0],
        profiles_root / "openvla-flappy" / "flappy" / "L_3090" / "e2e_latencies.jsonl",
    )
    exp1_rows = read_csv(tables_dir / "experiment1_latency_profile_summary.csv")

    with pytest.raises(FileNotFoundError, match="infer_latencies.jsonl"):
        bfd.build_latency_sample_rows(
            exp1_rows,
            profiles_root,
            panel_a_env="Flappy",
        )
