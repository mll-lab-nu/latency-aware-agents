from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

from latency_bench.latency.profile import write_latency_samples
from latency_bench.utils.io import read_csv, write_json, write_jsonl
from latency_bench.analysis import cross_device_figure_data as bfd
from latency_bench.analysis.cross_device_protocol_tables import write_protocol_tables


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _seed_tables(tables_dir: Path) -> None:
    _write_csv(
        tables_dir / "experiment1_latency_profile_summary.csv",
        ["experiment_id", "panel", "policy", "env", "profile_id", "profile_instance_id", "hardware", "env_fps",
         "inference_mean_ms", "e2e_mean_ms", "e2e_mean_frames", "n_samples", "notes"],
        [
            {"experiment_id": "exp1", "panel": "a", "policy": "openvla-flappy", "env": "Flappy",
             "profile_id": "L_3090", "profile_instance_id": "instance_rtx3090", "hardware": "RTX 3090", "env_fps": "30",
             "inference_mean_ms": "75.0", "e2e_mean_ms": "120.0", "e2e_mean_frames": "3.6",
             "n_samples": "70", "notes": ""},
            # Missing profile row (no mean_ms) -> dropped from table + no trace.
            {"experiment_id": "exp1", "panel": "a", "policy": "openvla-flappy", "env": "Flappy",
             "profile_id": "L_B200", "profile_instance_id": "", "hardware": "B200", "env_fps": "",
             "inference_mean_ms": "", "e2e_mean_ms": "", "e2e_mean_frames": "",
             "n_samples": "", "notes": "missing"},
        ],
    )
    _write_csv(
        tables_dir / "experiment2_same_device_calibration.csv",
        ["experiment_id", "panel", "policy", "env", "profile_id", "profile_instance_id", "hardware", "realtime_return",
         "simulated_return_using_L_d", "return_abs_error", "return_rel_error_pct", "realtime_length",
         "sim_length", "length_abs_error", "length_rel_error_pct", "n_episodes", "seed_set", "notes"],
        [
            {"experiment_id": "exp2", "panel": "b", "policy": "openvla-flappy", "env": "Flappy",
             "profile_id": "L_3090", "profile_instance_id": "instance_rtx3090", "hardware": "RTX 3090", "realtime_return": "434.5",
             "simulated_return_using_L_d": "444.6", "return_abs_error": "10.0", "return_rel_error_pct": "2.3",
             "realtime_length": "3519", "sim_length": "3600", "length_abs_error": "81", "length_rel_error_pct": "2.2",
             "n_episodes": "10", "seed_set": "0", "notes": ""},
        ],
    )
    _write_csv(
        tables_dir / "experiment3_cross_device_prediction.csv",
        ["experiment_id", "panel", "policy", "env", "target_device", "target_profile", "target_profile_instance_id", "replay_device",
         "replay_host_instance_id",
         "true_realtime_return_on_d", "simulated_return_on_replay_using_L_d", "return_abs_error",
         "return_rel_error_pct", "true_realtime_length_on_d", "simulated_length_on_replay_using_L_d",
         "length_abs_error", "length_rel_error_pct", "n_episodes", "seed_set", "use_in_main_claim", "notes"],
        [
            {"experiment_id": "exp3", "panel": "c", "policy": "openvla-flappy", "env": "Flappy",
             "target_device": "RTX 3090", "target_profile": "L_3090", "target_profile_instance_id": "instance_rtx3090",
             "replay_device": "B200", "replay_host_instance_id": "instance_b200",
             "true_realtime_return_on_d": "4.47", "simulated_return_on_replay_using_L_d": "3.9",
             "return_abs_error": "0.57", "return_rel_error_pct": "12.75", "true_realtime_length_on_d": "54.8",
             "simulated_length_on_replay_using_L_d": "50.0", "length_abs_error": "4.8", "length_rel_error_pct": "8.7",
             "n_episodes": "10", "seed_set": "0", "use_in_main_claim": "TRUE", "notes": ""},
        ],
    )


def _spec() -> dict:
    return {
        "protocol_id": "cross_device_latency_v1",
        "seed_set": [0],
        "devices": [
            {
                "id": "rtx3090",
                "display_name": "RTX 3090",
                "profile_id": "L_3090",
            },
        ],
        "rows": [
            {
                "policy": "openvla-flappy",
                "env": "Flappy",
                "model_type": "openvla",
                "env_name": "flappy",
            },
        ],
        "experiments": {
            "exp2_same_device_calibration": {"panel": "b"},
        },
    }


def _manifest(output_dir: Path, *, stage: str = "measure") -> dict:
    return {
        "protocol_id": "cross_device_latency_v1",
        "stage": stage,
        "status": "completed",
        "experiment_ids": [
            "exp1_latency_profile_summary",
            "exp2_same_device_calibration",
            "exp3_cross_device_prediction",
        ],
        "policy": "openvla-flappy",
        "env": "Flappy",
        "target_device_id": "rtx3090",
        "target_profile_id": "L_3090",
        "instance_id": "instance_rtx3090",
        "target_profile_instance_id": "instance_rtx3090",
        "replay_device_id": None,
        "seed": 0,
        "profile_path": str(output_dir.parents[3] / "profiles" / "openvla" / "flappy" / "rtx3090" / "instance_rtx3090" / "profile.json"),
        "output_dir": str(output_dir),
        "generated_config_path": str(output_dir / "config.yaml"),
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
            {"episode_id": 0, "episode_return_env": returns[0], "survival_steps": lengths[0], "metadata": {}},
            {"episode_id": 1, "episode_return_env": returns[1], "survival_steps": lengths[1], "metadata": {}},
        ],
    )


def test_build_all_produces_aligned_schemas(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    root = tmp_path / "cross_device"
    _seed_tables(tables_dir)
    # Panel-a traces for the one device that has a profile.
    profile_dir = root / "profiles" / "openvla" / "flappy" / "rtx3090" / "instance_rtx3090"
    write_jsonl(
        profile_dir / "e2e_latencies.jsonl",
        [{"latency_ms": float(i), "bench_overhead_ms": float(i) + 5.0} for i in range(100)],
    )
    write_latency_samples(
        [float(i + 100) for i in range(100)],
        profile_dir / "infer_latencies.jsonl",
    )
    (profile_dir / "profile.json").write_text("{}\n", encoding="utf-8")
    run_dir = root / "runs" / "measure" / "openvla-flappy" / "best_seed_0_n2"
    _write_episode_metrics(run_dir)
    sim_dir = root / "runs" / "sim_same" / "openvla-flappy" / "best_seed_0_n2"
    _write_episode_metrics(sim_dir, returns=(30.0, 40.0), lengths=(300, 400))
    write_jsonl(root / "manifests" / "runs.jsonl", [_manifest(run_dir), _manifest(sim_dir, stage="sim_same")])

    out_fig = tmp_path / "fig" / "data.csv"
    out_table = tmp_path / "table" / "data.csv"
    out_sampler_ablation = tmp_path / "fig" / "sampler_ablation.csv"
    out_sampler_ablation.parent.mkdir(parents=True)
    out_sampler_ablation.write_text("stale\n", encoding="utf-8")
    counts = bfd.build_all(
        tables_dir=tables_dir,
        out_fig=out_fig,
        out_table=out_table,
        out_sampler_ablation=out_sampler_ablation,
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
    assert table_rows[0]["profile_instance_id"] == "instance_rtx3090"
    assert table_rows[0]["inference_mean_ms"] == "75.0"
    assert table_rows[0]["e2e_mean_ms"] == "120.0"
    assert "p95_ms" not in table_rows[0]

    by_type: dict[str, list[dict[str, str]]] = {}
    for row in fig_rows:
        by_type.setdefault(row["row_type"], []).append(row)

    # Panel a: main figure samples e2e delay; inference samples are kept for the separate diagnostic figure.
    assert len(by_type["e2e_latency_sample"]) == 100
    assert len(by_type["overhead_latency_sample"]) == 100
    assert len(by_type["inference_latency_sample"]) == 100
    assert {r["hardware"] for r in by_type["e2e_latency_sample"]} == {"RTX 3090"}
    assert by_type["e2e_latency_sample"][0]["panel"] == "a"
    assert by_type["e2e_latency_sample"][0]["latency_ms"] == "0.0"
    assert by_type["inference_latency_sample"][0]["latency_ms"] == "100.0"

    # Panel b: calibration row passthrough.
    assert len(by_type["calibration"]) == 1
    assert by_type["calibration"][0]["policy"] == "openvla-flappy"
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

    assert counts == {"fig_rows": 306, "table_rows": 1, "sampler_ablation_rows": 0}
    assert read_csv(out_sampler_ablation) == []
    assert out_sampler_ablation.read_text(encoding="utf-8").splitlines() == [
        ",".join(bfd.SAMPLER_ABLATION_COLUMNS)
    ]


def test_realtime_episode_return_rows_read_hf_raw_layout(tmp_path: Path):
    raw_root = tmp_path / "raw"
    protocol_root = raw_root / "rtx3090" / "instances" / "instance_rtx3090" / "cross_device"
    old_root = Path("results/cross_device")
    run_dir = protocol_root / "runs" / "measure" / "openvla-flappy" / "best_seed_0_n2"
    _write_episode_metrics(run_dir)
    manifest = _manifest(old_root / run_dir.relative_to(protocol_root))
    write_jsonl(protocol_root / "manifests" / "runs.jsonl", [manifest])

    rows = bfd.build_episode_return_rows(
        _spec(),
        stage="measure",
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
    protocol_root = raw_root / "rtx3090" / "instances" / "instance_rtx3090" / "cross_device"
    old_root = Path("results/cross_device")
    run_dir = protocol_root / "runs" / "sim_same" / "openvla-flappy" / "best_seed_0_n2"
    _write_episode_metrics(run_dir, returns=(30.0, 40.0), lengths=(300, 400))
    manifest = _manifest(old_root / run_dir.relative_to(protocol_root), stage="sim_same")
    write_jsonl(protocol_root / "manifests" / "runs.jsonl", [manifest])

    rows = bfd.build_episode_return_rows(
        _spec(),
        stage="sim_same",
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
    root = tmp_path / "cross_device"
    _seed_tables(tables_dir)
    # Profiled measure run: e2e records carry bench_overhead_ms alongside latency_ms.
    profile_dir = root / "profiles" / "openvla" / "flappy" / "rtx3090" / "instance_rtx3090"
    write_jsonl(
        profile_dir / "e2e_latencies.jsonl",
        [{"latency_ms": float(i), "bench_overhead_ms": float(i) + 5.0} for i in range(100)],
    )
    write_latency_samples(
        [float(i + 100) for i in range(100)],
        profile_dir / "infer_latencies.jsonl",
    )
    (profile_dir / "profile.json").parent.mkdir(parents=True, exist_ok=True)
    (profile_dir / "profile.json").write_text("{}\n", encoding="utf-8")
    run_dir = root / "runs" / "measure" / "openvla-flappy" / "best_seed_0_n2"
    write_jsonl(root / "manifests" / "runs.jsonl", [_manifest(run_dir)])
    exp1_rows = read_csv(tables_dir / "experiment1_latency_profile_summary.csv")

    rows = bfd.build_latency_sample_rows(
        exp1_rows,
        panel_a_env="Flappy",
        spec=_spec(),
        root=root,
    )
    overhead = [row for row in rows if row["row_type"] == "overhead_latency_sample"]
    assert len(overhead) == 100
    assert {row["hardware"] for row in overhead} == {"RTX 3090"}
    assert overhead[0]["panel"] == "a"
    assert overhead[0]["latency_ms"] == 5.0  # bench_overhead_ms of sample 0


def test_build_latency_sample_rows_requires_inference_trace_for_profiled_devices(tmp_path: Path):
    tables_dir = tmp_path / "tables"
    root = tmp_path / "cross_device"
    _seed_tables(tables_dir)
    profile_dir = root / "profiles" / "openvla" / "flappy" / "rtx3090" / "instance_rtx3090"
    write_latency_samples(
        [1.0, 2.0],
        profile_dir / "e2e_latencies.jsonl",
    )
    (profile_dir / "profile.json").write_text("{}\n", encoding="utf-8")
    run_dir = root / "runs" / "measure" / "openvla-flappy" / "best_seed_0_n2"
    write_jsonl(root / "manifests" / "runs.jsonl", [_manifest(run_dir)])
    exp1_rows = read_csv(tables_dir / "experiment1_latency_profile_summary.csv")

    with pytest.raises(FileNotFoundError, match="infer_latencies.jsonl"):
        bfd.build_latency_sample_rows(
            exp1_rows,
            panel_a_env="Flappy",
            spec=_spec(),
            root=root,
        )


def _multi_session_spec() -> dict:
    spec = _spec()
    spec.update(
        {
            "protocol_id": "cross_device_10real_20sim_v1",
            "n_episodes": 2,
            "latency_seed_set": list(range(20)),
            "measure_sessions": {"fit": list(range(5)), "heldout": list(range(5, 10))},
            "latency_methods": ["temporal", "iid"],
            "primary_latency_method": "temporal",
        }
    )
    return spec


def _write_multi_session_protocol(tmp_path: Path) -> tuple[dict, Path, Path, list[float], dict[str, list[float]]]:
    spec = _multi_session_spec()
    root = tmp_path / "cross_device_10real_20sim_v1"
    raw_root = tmp_path / "hf" / "raw"
    build_id = "build_fit_0_4"
    profile_path = (
        root
        / "profile_builds"
        / "openvla"
        / "flappy"
        / "rtx3090"
        / "instance_rtx3090"
        / build_id
        / "profile.json"
    )
    write_json(
        profile_path,
        {
            "env_fps": 16.0,
            "n_samples": 2,
            "summary": {"mean_ms": 45.0, "mean_effective_frames": 0.72},
            "inference_summary": {"mean_ms": 32.5},
        },
    )
    write_jsonl(
        profile_path.parent / "e2e_latencies.jsonl",
        [
            {"latency_ms": 40.0, "bench_overhead_ms": 4.0},
            {"latency_ms": 50.0, "bench_overhead_ms": 5.0},
        ],
    )
    write_latency_samples([30.0, 35.0], profile_path.parent / "infer_latencies.jsonl")

    measure_root = (
        raw_root
        / "rtx3090"
        / "instances"
        / "instance_rtx3090"
        / "cross_device"
        / spec["protocol_id"]
    )
    fit_dirs: list[str] = []
    heldout_dirs: list[str] = []
    real_means = [10.0, 20.0, 30.0, 40.0, 50.0]
    for role, session_ids in spec["measure_sessions"].items():
        for session_id in session_ids:
            relative = f"runs/measure/session_{session_id}"
            run_dir = measure_root / relative
            mean = real_means[session_id - 5] if role == "heldout" else float(session_id)
            _write_episode_metrics(
                run_dir,
                returns=(mean - 1.0, mean + 1.0),
                lengths=(100 + session_id, 102 + session_id),
            )
            write_json(
                run_dir / "manifest.json",
                {
                    "status": "completed",
                    "protocol_id": spec["protocol_id"],
                    "policy": "openvla-flappy",
                    "target_device_id": "rtx3090",
                    "target_profile_instance_id": "instance_rtx3090",
                    "profile_build_id": build_id if role == "heldout" else None,
                    "seed": 0,
                    "measure_session_id": session_id,
                    "measure_role": role,
                    "n_episodes": spec["n_episodes"],
                },
            )
            (fit_dirs if role == "fit" else heldout_dirs).append(relative)
    measure_completion = {
        "protocol_id": spec["protocol_id"],
        "policy": "openvla-flappy",
        "model_type": "openvla",
        "env_name": "flappy",
        "target_device_id": "rtx3090",
        "target_profile_instance_id": "instance_rtx3090",
        "profile_build_id": build_id,
        "profile_artifact_sha256": "artifact_a",
        "profile_fitter_sha256": "fitter_a",
        "seed_set": spec["seed_set"],
        "n_episodes": spec["n_episodes"],
        "measure_sessions": spec["measure_sessions"],
        "profile_path": str(profile_path.relative_to(root)),
        "fit_run_dirs": fit_dirs,
        "heldout_run_dirs": heldout_dirs,
    }
    write_json(
        measure_root
        / "completion"
        / "measure"
        / "rtx3090"
        / "instance_rtx3090"
        / f"{build_id}.json",
        measure_completion,
    )

    sim_means = {
        "temporal": [float(value) for value in range(100, 120)],
        "iid": [float(value) for value in range(200, 220)],
    }
    for method, means in sim_means.items():
        run_dirs = []
        for latency_seed, mean in enumerate(means):
            relative = f"runs/sim_cross/{method}/latency_seed_{latency_seed}"
            run_dir = root / relative
            _write_episode_metrics(
                run_dir,
                returns=(mean - 2.0, mean + 2.0),
                lengths=(200 + latency_seed, 204 + latency_seed),
            )
            write_json(
                run_dir / "manifest.json",
                {
                    "status": "completed",
                    "protocol_id": spec["protocol_id"],
                    "policy": "openvla-flappy",
                    "target_device_id": "rtx3090",
                    "target_profile_instance_id": "instance_rtx3090",
                    "profile_build_id": build_id,
                    "latency_seed": latency_seed,
                    "latency_method": method,
                    "sim_host_device_id": "rtx5090",
                    "sim_host_instance_id": "instance_5090",
                    "seed": 0,
                    "n_episodes": spec["n_episodes"],
                },
            )
            run_dirs.append(relative)
        write_json(
            root
            / "completion"
            / "sim"
            / method
            / "rtx5090"
            / "instance_5090"
            / "rtx3090"
            / "instance_rtx3090"
            / f"{build_id}.json",
            {
                "protocol_id": spec["protocol_id"],
                "policy": "openvla-flappy",
                "latency_method": method,
                "sim_host_device_id": "rtx5090",
                "sim_host_instance_id": "instance_5090",
                "target_device_id": "rtx3090",
                "target_profile_instance_id": "instance_rtx3090",
                "profile_build_id": build_id,
                "profile_artifact_sha256": "artifact_a",
                "profile_fitter_sha256": "fitter_a",
                "seed_set": spec["seed_set"],
                "n_episodes": spec["n_episodes"],
                "latency_seed_set": spec["latency_seed_set"],
                "run_dirs": run_dirs,
            },
        )
    return spec, root, raw_root, real_means, sim_means


def test_multi_session_figure_data_uses_completion_scoped_session_statistics(tmp_path: Path):
    spec, root, raw_root, real_means, sim_means = _write_multi_session_protocol(tmp_path)
    tables_dir = tmp_path / "tables"
    write_protocol_tables(
        spec=spec,
        root=root,
        raw_root=raw_root,
        out=tables_dir,
        overwrite=True,
    )

    experiment2 = read_csv(tables_dir / "experiment2_same_device_calibration.csv")
    assert {row["latency_method"] for row in experiment2} == {"temporal", "iid"}
    temporal = next(row for row in experiment2 if row["latency_method"] == "temporal")
    assert float(temporal["realtime_return"]) == pytest.approx(30.0)
    assert float(temporal["simulated_return_using_L_d"]) == pytest.approx(109.5)
    assert temporal["realtime_session_count"] == "5"
    assert temporal["simulated_session_count"] == "20"
    assert temporal["n_episodes"] == "2"
    assert temporal["realtime_n_episodes"] == "10"
    assert temporal["simulated_n_episodes"] == "40"
    real_std = float(np.std(real_means, ddof=1))
    assert float(temporal["realtime_return_ci95"]) == pytest.approx(
        student_t.ppf(0.975, 4) * real_std / np.sqrt(5)
    )

    out_fig = tmp_path / "figure" / "data.csv"
    out_ablation = tmp_path / "figure" / "sampler_ablation.csv"
    counts = bfd.build_all(
        tables_dir=tables_dir,
        out_fig=out_fig,
        out_table=tmp_path / "table" / "data.csv",
        out_sampler_ablation=out_ablation,
        panel_a_env="Flappy",
        spec=spec,
        root=root,
        raw_root=raw_root,
    )

    figure_rows = read_csv(out_fig)
    by_type: dict[str, list[dict[str, str]]] = {}
    for row in figure_rows:
        by_type.setdefault(row["row_type"], []).append(row)
    assert counts == {"fig_rows": 7, "table_rows": 1, "sampler_ablation_rows": 2}
    assert len(by_type["e2e_latency_sample"]) == 2
    assert len(by_type["overhead_latency_sample"]) == 2
    assert len(by_type["inference_latency_sample"]) == 2
    assert {row["profile_build_id"] for row in by_type["e2e_latency_sample"]} == {
        "build_fit_0_4"
    }
    assert len(by_type["calibration"]) == 1
    assert by_type["calibration"][0]["latency_method"] == "temporal"
    assert "prediction" not in by_type

    ablation = read_csv(out_ablation)
    assert {row["latency_method"] for row in ablation} == {"temporal", "iid"}
    temporal_ablation = next(row for row in ablation if row["latency_method"] == "temporal")
    assert float(temporal_ablation["sim_minus_real_return"]) == pytest.approx(79.5)
    real_term = np.var(real_means, ddof=1) / len(real_means)
    temporal_term = np.var(sim_means["temporal"], ddof=1) / len(sim_means["temporal"])
    degrees_of_freedom = (real_term + temporal_term) ** 2 / (
        real_term**2 / 4 + temporal_term**2 / 19
    )
    assert float(temporal_ablation["sim_minus_real_ci95"]) == pytest.approx(
        student_t.ppf(0.975, degrees_of_freedom) * np.sqrt(real_term + temporal_term)
    )


def test_multi_session_tables_require_sim_completion(tmp_path: Path):
    spec, root, raw_root, _, _ = _write_multi_session_protocol(tmp_path)
    for completion in (root / "completion" / "sim").glob("**/*.json"):
        completion.unlink()

    write_protocol_tables(
        spec=spec,
        root=root,
        raw_root=raw_root,
        out=tmp_path / "tables",
        overwrite=True,
    )

    assert read_csv(tmp_path / "tables" / "experiment2_same_device_calibration.csv") == []


def test_multi_session_completion_fails_when_a_referenced_run_is_missing(tmp_path: Path):
    spec, root, raw_root, _, _ = _write_multi_session_protocol(tmp_path)
    missing = root / "runs" / "sim_cross" / "temporal" / "latency_seed_19" / "manifest.json"
    missing.unlink()

    with pytest.raises(FileNotFoundError):
        write_protocol_tables(
            spec=spec,
            root=root,
            raw_root=raw_root,
            out=tmp_path / "tables",
            overwrite=True,
        )


def test_multi_session_completion_must_match_protocol_seed_set(tmp_path: Path):
    spec, root, raw_root, _, _ = _write_multi_session_protocol(tmp_path)
    completion_path = next((root / "completion" / "sim" / "temporal").glob("**/*.json"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["latency_seed_set"] = [0]
    write_json(completion_path, completion)

    with pytest.raises(ValueError, match="sim completion"):
        write_protocol_tables(
            spec=spec,
            root=root,
            raw_root=raw_root,
            out=tmp_path / "tables",
            overwrite=True,
        )


def test_multi_session_completion_must_match_profile_artifact(tmp_path: Path):
    spec, root, raw_root, _, _ = _write_multi_session_protocol(tmp_path)
    completion_path = next((root / "completion" / "sim" / "iid").glob("**/*.json"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["profile_artifact_sha256"] = "different_artifact"
    write_json(completion_path, completion)

    with pytest.raises(ValueError, match="measured profile artifact"):
        write_protocol_tables(
            spec=spec,
            root=root,
            raw_root=raw_root,
            out=tmp_path / "tables",
            overwrite=True,
        )


def test_multi_session_completion_checks_actual_episode_count(tmp_path: Path):
    spec, root, raw_root, _, _ = _write_multi_session_protocol(tmp_path)
    metrics = root / "runs/sim_cross/temporal/latency_seed_19/episode_metrics.jsonl"
    write_jsonl(
        metrics,
        [{"episode_id": 0, "episode_return_env": 100.0, "survival_steps": 100, "metadata": {}}],
    )

    with pytest.raises(ValueError, match="simulation runs"):
        write_protocol_tables(
            spec=spec,
            root=root,
            raw_root=raw_root,
            out=tmp_path / "tables",
            overwrite=True,
        )
