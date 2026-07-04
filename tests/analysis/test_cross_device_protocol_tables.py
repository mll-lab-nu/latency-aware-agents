from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from latency_bench.analysis.cross_device_protocol_tables import (
    EXPERIMENT1_COLUMNS,
    EXPERIMENT2_COLUMNS,
    EXPERIMENT3_COLUMNS,
    build_experiment1_rows,
    build_experiment2_rows,
    build_experiment3_rows,
    write_protocol_tables,
)
from latency_bench.experiments.build_latency_profile_from_trace import build_profile_from_trace
from latency_bench.latency.profile import write_latency_samples
from latency_bench.utils.io import write_jsonl


def _spec() -> dict:
    row = {
        "policy": "openvla-flappy",
        "env": "Flappy",
        "policy_type": "starvla",
        "model_type": "openvla",
        "env_name": "flappy",
        "fps": 30,
        "base_config": "unused.yaml",
    }
    return {
        "protocol_id": "cross_device_latency_v1",
        "seed_set": [0],
        "devices": [
            {
                "id": "a100",
                "display_name": "A100",
                "profile_id": "L_A100",
                "aliases": ["NVIDIA A100"],
            },
            {
                "id": "h200",
                "display_name": "H200",
                "profile_id": "L_H200",
                "aliases": ["NVIDIA H200"],
            },
        ],
        "experiments": {
            "exp1_latency_profile_summary": {"panel": "a", "rows": [dict(row)]},
            "exp2_same_device_calibration": {"panel": "b", "rows": [dict(row)]},
            "exp3_cross_device_prediction": {
                "panel": "c",
                "rows": [dict(row)],
                "replay_pairs": [
                    {"target_device": "a100", "replay_device": "h200"},
                    {"target_device": "h200", "replay_device": "a100"},
                ],
            },
        },
    }


def _build_profile(root: Path, *, device_id: str = "a100", profile_id: str = "L_A100") -> Path:
    raw_infer = root / "raw_infer_latencies.jsonl"
    raw_e2e = root / "raw_e2e_latencies.jsonl"
    write_latency_samples([10.0, 20.0, 30.0], raw_infer)
    write_latency_samples([20.0, 40.0, 60.0], raw_e2e)
    instance_id = f"instance_{device_id}"
    profile_dir = root / "profiles" / "openvla" / "flappy" / device_id / instance_id
    build_profile_from_trace(
        latencies=raw_e2e,
        output_dir=profile_dir,
        profile_id=profile_id,
        policy_id="openvla-flappy",
        env_id="flappy",
        target_device=device_id,
        env_fps=16.0,
        metadata={},
        latency_kind="e2e_latency",
        inference_latencies=raw_infer,
    )
    return profile_dir / "profile.json"


def _episode_records(mean_return: float, mean_length: float) -> list[dict]:
    return [
        {"episode_return_env": mean_return - 5.0, "survival_steps": mean_length - 50.0},
        {"episode_return_env": mean_return + 5.0, "survival_steps": mean_length + 50.0},
    ]


EXP_ALL = [
    "exp1_latency_profile_summary",
    "exp2_same_device_calibration",
    "exp3_cross_device_prediction",
]


def _manifest(
    *,
    stage: str,
    experiment_id: str,
    target_device_id: str,
    output_dir: Path,
    experiment_ids: list[str] | None = None,
    replay_device_id: str | None = None,
    instance_id: str | None = "instance_a100",
    target_profile_instance_id: str | None = None,
    profile_path: Path | None = None,
    status: str = "completed",
    skip_reason: str = "",
) -> dict:
    return {
        "stage": stage,
        "experiment_id": experiment_id,
        "experiment_ids": experiment_ids if experiment_ids is not None else [experiment_id],
        "policy": "openvla-flappy",
        "env": "Flappy",
        "target_device_id": target_device_id,
        "target_profile_id": f"L_{target_device_id.upper()}",
        "instance_id": instance_id,
        "target_profile_instance_id": target_profile_instance_id or instance_id,
        "replay_device_id": replay_device_id,
        "status": status,
        "skip_reason": skip_reason,
        "notes": skip_reason,
        "output_dir": str(output_dir),
        "profile_path": "" if profile_path is None else str(profile_path),
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
    }


def _write_fixture(root: Path) -> None:
    profile_path = _build_profile(root)
    h200_profile_path = _build_profile(root, device_id="h200", profile_id="L_H200")
    # measure on a device IS its realtime run: episode_metrics here = realtime return.
    measure_a100 = root / "runs" / "measure" / "openvla-flappy" / "flappy" / "a100" / "seed_0"
    measure_h200 = root / "runs" / "measure" / "openvla-flappy" / "flappy" / "h200" / "seed_0"
    sim_same = root / "runs" / "sim_same" / "openvla-flappy" / "flappy" / "a100" / "seed_0"
    sim_cross = (
        root
        / "runs"
        / "sim_cross"
        / "openvla-flappy"
        / "flappy"
        / "a100"
        / "replay_h200"
        / "seed_0"
    )
    write_jsonl(measure_a100 / "episode_metrics.jsonl", _episode_records(100.0, 1000.0))
    write_jsonl(sim_same / "episode_metrics.jsonl", _episode_records(110.0, 1100.0))
    write_jsonl(sim_cross / "episode_metrics.jsonl", _episode_records(90.0, 950.0))
    manifests = [
        # a100 measure contributes to all experiments (profile + realtime return).
        _manifest(
            stage="measure",
            experiment_id="exp1_latency_profile_summary",
            experiment_ids=EXP_ALL,
            target_device_id="a100",
            output_dir=measure_a100,
            profile_path=profile_path,
        ),
        # h200 measure provides only its profile (no return run): exp3 h200-target stays "missing".
        _manifest(
            stage="measure",
            experiment_id="exp1_latency_profile_summary",
            experiment_ids=["exp1_latency_profile_summary"],
            target_device_id="h200",
            instance_id="instance_h200",
            output_dir=measure_h200,
            profile_path=h200_profile_path,
        ),
        _manifest(stage="sim_same", experiment_id="exp2_same_device_calibration", target_device_id="a100", output_dir=sim_same),
        _manifest(
            stage="sim_cross",
            experiment_id="exp3_cross_device_prediction",
            target_device_id="a100",
            target_profile_instance_id="instance_a100",
            replay_device_id="h200",
            instance_id="instance_h200",
            output_dir=sim_cross,
        ),
    ]
    write_jsonl(root / "manifests" / "runs.jsonl", manifests)


def _write_raw_download_fixture(protocol_root: Path, *, include_skipped_duplicates: bool = False) -> None:
    profile_path = _build_profile(protocol_root)
    h200_profile_path = _build_profile(protocol_root, device_id="h200", profile_id="L_H200")
    measure_a100 = protocol_root / "runs" / "measure" / "openvla-flappy" / "flappy" / "a100" / "seed_0"
    measure_h200 = protocol_root / "runs" / "measure" / "openvla-flappy" / "flappy" / "h200" / "seed_0"
    sim_same = protocol_root / "runs" / "sim_same" / "openvla-flappy" / "flappy" / "a100" / "seed_0"
    old_root = Path("results/cross_device")
    write_jsonl(measure_a100 / "episode_metrics.jsonl", _episode_records(100.0, 1000.0))
    write_jsonl(sim_same / "episode_metrics.jsonl", _episode_records(110.0, 1100.0))
    manifests = [
        _manifest(
            stage="measure",
            experiment_id="exp1_latency_profile_summary",
            experiment_ids=EXP_ALL,
            target_device_id="a100",
            output_dir=old_root / measure_a100.relative_to(protocol_root),
            profile_path=old_root / profile_path.relative_to(protocol_root),
        ),
        _manifest(
            stage="measure",
            experiment_id="exp1_latency_profile_summary",
            experiment_ids=["exp1_latency_profile_summary"],
            target_device_id="h200",
            output_dir=old_root / measure_h200.relative_to(protocol_root),
            profile_path=old_root / h200_profile_path.relative_to(protocol_root),
        ),
        _manifest(
            stage="sim_same",
            experiment_id="exp2_same_device_calibration",
            target_device_id="a100",
            output_dir=old_root / sim_same.relative_to(protocol_root),
        ),
    ]
    if include_skipped_duplicates:
        manifests.extend(
            [
                _manifest(
                    stage="measure",
                    experiment_id="exp1_latency_profile_summary",
                    experiment_ids=EXP_ALL,
                    target_device_id="a100",
                    output_dir=old_root / measure_a100.relative_to(protocol_root),
                    status="skipped",
                    skip_reason="runtime_env selection mismatch: selected openvla, row requires latency",
                ),
                _manifest(
                    stage="sim_same",
                    experiment_id="exp2_same_device_calibration",
                    target_device_id="a100",
                    output_dir=old_root / sim_same.relative_to(protocol_root),
                    status="skipped",
                    skip_reason="runtime_env selection mismatch: selected openvla, row requires latency",
                ),
            ]
        )
    write_jsonl(protocol_root / "manifests" / "runs.jsonl", manifests)


def test_profile_summary_mapping_and_exact_error_formulas(tmp_path: Path):
    spec = _spec()
    _write_fixture(tmp_path)

    exp1 = build_experiment1_rows(spec, tmp_path)
    a100_profile = next(row for row in exp1 if row["profile_id"] == "L_A100")
    assert tuple(a100_profile) == EXPERIMENT1_COLUMNS
    assert a100_profile["profile_instance_id"] == "instance_a100"
    assert float(a100_profile["inference_mean_ms"]) == pytest.approx(20.0)
    assert float(a100_profile["e2e_mean_ms"]) == pytest.approx(40.0)
    assert float(a100_profile["e2e_mean_frames"]) == pytest.approx(0.64)
    assert int(a100_profile["n_samples"]) == 3
    assert float(a100_profile["env_fps"]) == pytest.approx(16.0)

    exp2 = build_experiment2_rows(spec, tmp_path)
    a100_calibration = next(row for row in exp2 if row["hardware"] == "A100")
    assert tuple(a100_calibration) == EXPERIMENT2_COLUMNS
    assert a100_calibration["profile_instance_id"] == "instance_a100"
    assert float(a100_calibration["realtime_return"]) == pytest.approx(100.0)
    assert float(a100_calibration["simulated_return_using_L_d"]) == pytest.approx(110.0)
    assert float(a100_calibration["return_abs_error"]) == pytest.approx(10.0)
    assert float(a100_calibration["return_rel_error_pct"]) == pytest.approx(10.0)
    assert float(a100_calibration["realtime_length"]) == pytest.approx(1000.0)
    assert float(a100_calibration["sim_length"]) == pytest.approx(1100.0)
    assert float(a100_calibration["length_abs_error"]) == pytest.approx(100.0)
    assert float(a100_calibration["length_rel_error_pct"]) == pytest.approx(10.0)


def test_experiment1_requires_profile_env_fps(tmp_path: Path):
    spec = _spec()
    profile_path = _build_profile(tmp_path)
    measure_a100 = tmp_path / "runs" / "measure" / "openvla-flappy" / "flappy" / "a100" / "seed_0"
    write_jsonl(measure_a100 / "episode_metrics.jsonl", _episode_records(100.0, 1000.0))
    write_jsonl(
        tmp_path / "manifests" / "runs.jsonl",
        [
            _manifest(
                stage="measure",
                experiment_id="exp1_latency_profile_summary",
                target_device_id="a100",
                output_dir=measure_a100,
                profile_path=profile_path,
            )
        ],
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    del profile["env_fps"]
    profile_path.write_text(json.dumps(profile, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(KeyError):
        build_experiment1_rows(spec, tmp_path)


def test_experiment1_requires_inference_and_e2e_profile_summaries(tmp_path: Path):
    spec = _spec()
    profile_path = _build_profile(tmp_path)
    measure_a100 = tmp_path / "runs" / "measure" / "openvla-flappy" / "flappy" / "a100" / "seed_0"
    write_jsonl(measure_a100 / "episode_metrics.jsonl", _episode_records(100.0, 1000.0))
    write_jsonl(
        tmp_path / "manifests" / "runs.jsonl",
        [
            _manifest(
                stage="measure",
                experiment_id="exp1_latency_profile_summary",
                target_device_id="a100",
                output_dir=measure_a100,
                profile_path=profile_path,
            )
        ],
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    del profile["inference_summary"]
    profile_path.write_text(json.dumps(profile, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(KeyError, match="inference_summary"):
        build_experiment1_rows(spec, tmp_path)


def test_manifest_matching_and_main_claim_rule(tmp_path: Path):
    spec = _spec()
    _write_fixture(tmp_path)

    exp3 = build_experiment3_rows(spec, tmp_path)
    a100_on_h200 = next(row for row in exp3 if row["target_device"] == "A100" and row["replay_device"] == "H200")
    assert tuple(a100_on_h200) == EXPERIMENT3_COLUMNS
    assert a100_on_h200["target_profile_instance_id"] == "instance_a100"
    assert a100_on_h200["replay_host_instance_id"] == "instance_h200"
    assert a100_on_h200["use_in_main_claim"] == "TRUE"
    assert float(a100_on_h200["true_realtime_return_on_d"]) == pytest.approx(100.0)
    assert float(a100_on_h200["simulated_return_on_replay_using_L_d"]) == pytest.approx(90.0)
    assert float(a100_on_h200["return_rel_error_pct"]) == pytest.approx(10.0)
    assert a100_on_h200["notes"] == ""

    h200_on_a100 = next(row for row in exp3 if row["target_device"] == "H200" and row["replay_device"] == "A100")
    assert h200_on_a100["use_in_main_claim"] == "TRUE"
    assert "missing true realtime run" in h200_on_a100["notes"]
    assert "missing sim-cross run" in h200_on_a100["notes"]


def test_same_device_rows_do_not_merge_instances_with_same_hardware_class(tmp_path: Path):
    spec = _spec()
    first_profile = _build_profile(tmp_path)
    second_profile = _build_profile(tmp_path / "second")
    second_profile_dir = tmp_path / "profiles" / "openvla" / "flappy" / "a100" / "instance_second"
    second_profile_dir.mkdir(parents=True)
    second_profile.rename(second_profile_dir / "profile.json")

    measure_first = tmp_path / "runs" / "measure" / "openvla-flappy" / "flappy" / "a100" / "first"
    measure_second = tmp_path / "runs" / "measure" / "openvla-flappy" / "flappy" / "a100" / "second"
    sim_first = tmp_path / "runs" / "sim_same" / "openvla-flappy" / "flappy" / "a100" / "first"
    sim_second = tmp_path / "runs" / "sim_same" / "openvla-flappy" / "flappy" / "a100" / "second"
    write_jsonl(measure_first / "episode_metrics.jsonl", _episode_records(100.0, 1000.0))
    write_jsonl(measure_second / "episode_metrics.jsonl", _episode_records(200.0, 2000.0))
    write_jsonl(sim_first / "episode_metrics.jsonl", _episode_records(110.0, 1100.0))
    write_jsonl(sim_second / "episode_metrics.jsonl", _episode_records(210.0, 2100.0))
    write_jsonl(
        tmp_path / "manifests" / "runs.jsonl",
        [
            _manifest(
                stage="measure",
                experiment_id="exp2_same_device_calibration",
                target_device_id="a100",
                instance_id="instance_a100",
                output_dir=measure_first,
                profile_path=first_profile,
            ),
            _manifest(
                stage="measure",
                experiment_id="exp2_same_device_calibration",
                target_device_id="a100",
                instance_id="instance_second",
                output_dir=measure_second,
                profile_path=second_profile_dir / "profile.json",
            ),
            _manifest(
                stage="sim_same",
                experiment_id="exp2_same_device_calibration",
                target_device_id="a100",
                instance_id="instance_a100",
                output_dir=sim_first,
            ),
            _manifest(
                stage="sim_same",
                experiment_id="exp2_same_device_calibration",
                target_device_id="a100",
                instance_id="instance_second",
                output_dir=sim_second,
            ),
        ],
    )

    rows = [row for row in build_experiment2_rows(spec, tmp_path) if row["hardware"] == "A100"]

    assert [row["profile_instance_id"] for row in rows] == ["instance_a100", "instance_second"]
    assert [float(row["realtime_return"]) for row in rows] == [100.0, 200.0]
    assert [float(row["simulated_return_using_L_d"]) for row in rows] == [110.0, 210.0]


def test_write_protocol_tables_uses_requested_schemas(tmp_path: Path):
    spec = _spec()
    _write_fixture(tmp_path)
    output_dir = tmp_path / "tables"

    outputs = write_protocol_tables(spec=spec, root=tmp_path, out=output_dir)

    assert set(outputs) == {"experiment1", "experiment2", "experiment3"}
    with outputs["experiment1"].open("r", encoding="utf-8", newline="") as f:
        assert tuple(csv.DictReader(f).fieldnames or []) == EXPERIMENT1_COLUMNS
    with outputs["experiment2"].open("r", encoding="utf-8", newline="") as f:
        assert tuple(csv.DictReader(f).fieldnames or []) == EXPERIMENT2_COLUMNS
    with outputs["experiment3"].open("r", encoding="utf-8", newline="") as f:
        assert tuple(csv.DictReader(f).fieldnames or []) == EXPERIMENT3_COLUMNS

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_protocol_tables(spec=spec, root=tmp_path, out=output_dir)
    write_protocol_tables(spec=spec, root=tmp_path, out=output_dir, overwrite=True)

    malformed = _spec()
    del malformed["experiments"]
    with pytest.raises(KeyError, match="experiments"):
        write_protocol_tables(spec=malformed, root=tmp_path, out=tmp_path / "bad_tables")


def test_write_protocol_tables_reads_hf_raw_download_layout(tmp_path: Path):
    spec = _spec()
    raw_root = tmp_path / "results" / "raw" / "raw"
    protocol_root = raw_root / "a100" / "combined" / "host_20260101_000000" / "cross_device"
    standard_root = tmp_path / "results" / "cross_device"
    output_dir = standard_root / "tables"
    _write_raw_download_fixture(protocol_root)

    outputs = write_protocol_tables(spec=spec, root=standard_root, raw_root=raw_root, out=output_dir)

    with outputs["experiment1"].open("r", encoding="utf-8", newline="") as f:
        exp1 = list(csv.DictReader(f))
    a100_profile = next(row for row in exp1 if row["profile_id"] == "L_A100")
    assert float(a100_profile["inference_mean_ms"]) == pytest.approx(20.0)
    assert float(a100_profile["e2e_mean_ms"]) == pytest.approx(40.0)
    assert int(a100_profile["n_samples"]) == 3

    with outputs["experiment2"].open("r", encoding="utf-8", newline="") as f:
        exp2 = list(csv.DictReader(f))
    a100_calibration = next(row for row in exp2 if row["hardware"] == "A100")
    assert float(a100_calibration["realtime_return"]) == pytest.approx(100.0)
    assert float(a100_calibration["simulated_return_using_L_d"]) == pytest.approx(110.0)
    assert a100_calibration["notes"] == ""


def test_write_protocol_tables_reads_direct_results_raw_layout(tmp_path: Path):
    spec = _spec()
    raw_root = tmp_path / "results" / "raw"
    protocol_root = raw_root / "a100" / "combined" / "host_20260101_000000" / "cross_device"
    standard_root = tmp_path / "results" / "cross_device"
    output_dir = standard_root / "tables"
    _write_raw_download_fixture(protocol_root)

    outputs = write_protocol_tables(spec=spec, root=standard_root, raw_root=raw_root, out=output_dir)

    with outputs["experiment2"].open("r", encoding="utf-8", newline="") as f:
        exp2 = list(csv.DictReader(f))
    a100_calibration = next(row for row in exp2 if row["hardware"] == "A100")
    assert float(a100_calibration["realtime_return"]) == pytest.approx(100.0)
    assert float(a100_calibration["simulated_return_using_L_d"]) == pytest.approx(110.0)


def test_completed_manifest_wins_over_later_skipped_duplicate(tmp_path: Path):
    spec = _spec()
    raw_root = tmp_path / "results" / "raw"
    protocol_root = raw_root / "a100" / "combined" / "host_20260101_000000" / "cross_device"
    standard_root = tmp_path / "results" / "cross_device"
    _write_raw_download_fixture(protocol_root, include_skipped_duplicates=True)

    exp2 = build_experiment2_rows(spec, standard_root, raw_root=raw_root)

    a100_calibration = next(row for row in exp2 if row["hardware"] == "A100")
    assert float(a100_calibration["realtime_return"]) == pytest.approx(100.0)
    assert float(a100_calibration["simulated_return_using_L_d"]) == pytest.approx(110.0)
    assert "runtime_env selection mismatch" not in a100_calibration["notes"]


def test_aggregate_notes_include_skipped_manifest_reasons(tmp_path: Path):
    spec = _spec()
    sim_same = tmp_path / "runs" / "sim_same" / "openvla-flappy" / "flappy" / "a100" / "seed_0"
    manifests = [
        _manifest(
            stage="sim_same",
            experiment_id="exp2_same_device_calibration",
            target_device_id="a100",
            output_dir=sim_same,
            status="skipped",
            skip_reason="runtime_env mismatch: requires latency",
        )
    ]
    write_jsonl(tmp_path / "manifests" / "runs.jsonl", manifests)

    exp2 = build_experiment2_rows(spec, tmp_path)
    a100 = next(row for row in exp2 if row["hardware"] == "A100")

    assert a100["realtime_return"] == ""
    assert "missing realtime run" in a100["notes"]
    assert "runtime_env mismatch" in a100["notes"]
