from __future__ import annotations

from pathlib import Path

import pytest

from latency_bench.analysis.cross_device_results import write_study_outputs
from latency_bench.analysis.profile_stats import compute_latency_profile_summary
from latency_bench.cross_device.spec import load_study
from latency_bench.latency.profile import write_latency_samples
from latency_bench.utils.io import read_csv, write_json, write_jsonl


def _study(tmp_path: Path, name: str) -> dict:
    study = load_study(Path("configs/experiments") / name)
    study["devices"] = [study["devices"][0]]
    study["latency_seeds"] = [0, 1, 2]
    study["output_root"] = str(tmp_path / "local")
    return study


def _write_run(path: Path, return_value: float, length_value: float) -> None:
    write_jsonl(
        path / "episode_metrics.jsonl",
        [
            {
                "episode_return_env": return_value,
                "survival_steps": length_value,
                "metadata": {},
            },
            {
                "episode_return_env": return_value,
                "survival_steps": length_value,
                "metadata": {},
            },
        ],
    )
    (path / "DONE").touch()


def _write_measurement(
    root: Path,
    study: dict,
    instance_id: str,
    real_values: dict[int, tuple[float, float]],
    samples: list[float],
) -> None:
    device = study["devices"][0]
    measure_dir = root / "measure" / device["id"] / instance_id
    for session_id, (return_value, length_value) in real_values.items():
        _write_run(measure_dir / "sessions" / str(session_id), return_value, length_value)
    for session_id in study["sessions"]["fit"]:
        session_dir = measure_dir / "sessions" / str(session_id)
        write_jsonl(
            session_dir / "e2e_latencies.jsonl",
            [
                {"latency_ms": sample, "bench_overhead_ms": sample / 10.0}
                for sample in samples
            ],
        )
        (session_dir / "DONE").touch()
    profile_dir = measure_dir / "profile"
    summary = compute_latency_profile_summary(samples, env_fps=10.0)
    inference_samples = [sample / 2.0 for sample in samples]
    write_latency_samples(samples, profile_dir / "e2e_latencies.jsonl")
    write_latency_samples(inference_samples, profile_dir / "infer_latencies.jsonl")
    write_json(
        profile_dir / "profile.json",
        {
            "profile_id": device["profile_id"],
            "env_fps": 10.0,
            "frame_ms": 100.0,
            "trace_path": "e2e_latencies.jsonl",
            "inference_trace_path": "infer_latencies.jsonl",
            "latency_kind": "e2e_latency",
            "summary": summary,
            "inference_summary": compute_latency_profile_summary(inference_samples, env_fps=10.0),
        },
    )
    (profile_dir / "DONE").touch()
    (measure_dir / "DONE").touch()


def _write_simulations(
    root: Path,
    study: dict,
    instance_id: str,
    method_values: dict[str, list[tuple[float, float]]],
) -> None:
    device_id = study["devices"][0]["id"]
    for method, values in method_values.items():
        method_dir = root / "sim" / device_id / instance_id / method
        for seed, (return_value, length_value) in zip(study["latency_seeds"], values):
            _write_run(method_dir / str(seed), return_value, length_value)
        (method_dir / "DONE").touch()


def test_real1_outputs_in_sample_rows_with_empty_real_intervals(tmp_path):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    root = tmp_path / "study"
    _write_measurement(root, study, "instance_a", {0: (2.0, 20.0)}, [10.0, 20.0, 30.0])
    _write_simulations(
        root,
        study,
        "instance_a",
        {
            "temporal": [(1.0, 10.0), (3.0, 30.0), (5.0, 50.0)],
            "iid": [(2.0, 20.0), (4.0, 40.0), (6.0, 60.0)],
        },
    )

    paths = write_study_outputs(study, root, tmp_path / "analysis")

    samples = read_csv(paths["latency_samples"])
    profiles = read_csv(paths["latency_profile_summary"])
    rows = read_csv(paths["real_vs_sim"])
    temporal_return = next(
        row
        for row in rows
        if row["latency_method"] == "temporal" and row["metric"] == "episode_return_env"
    )
    assert [
        float(row["latency_ms"])
        for row in samples
        if row["latency_kind"] == "e2e_latency"
    ] == [10.0, 20.0, 30.0]
    assert {row["latency_kind"] for row in samples} == {
        "e2e_latency",
        "inference_latency",
        "overhead_latency",
    }
    assert [
        float(row["latency_ms"])
        for row in samples
        if row["latency_kind"] == "overhead_latency"
    ] == [1.0, 2.0, 3.0]
    assert float(profiles[0]["mean_ms"]) == pytest.approx(20.0)
    assert float(profiles[0]["inference_mean_ms"]) == pytest.approx(10.0)
    assert temporal_return["evaluation_scope"] == "in_sample"
    assert temporal_return["real_session_count"] == "1"
    assert temporal_return["real_episode_count"] == "2"
    assert float(temporal_return["real_mean"]) == pytest.approx(2.0)
    assert temporal_return["real_std"] == ""
    assert temporal_return["real_ci95"] == ""
    assert float(temporal_return["sim_mean"]) == pytest.approx(3.0)
    assert float(temporal_return["sim_minus_real"]) == pytest.approx(1.0)
    assert temporal_return["difference_ci95"] == ""
    assert {row["latency_method"] for row in rows} == {"temporal", "iid"}
    assert all("sim_host_device_id" not in row for row in rows)


def test_real10_uses_heldout_session_and_seed_means_per_instance(tmp_path):
    study = _study(tmp_path, "cross_device_10real_20sim.yaml")
    root = tmp_path / "study"
    real_values = {
        session_id: (float(index), float(index * 10))
        for index, session_id in enumerate(study["sessions"]["evaluation"], start=1)
    }
    methods = {
        "temporal": [(2.0, 20.0), (4.0, 40.0), (6.0, 60.0)],
        "iid": [(3.0, 30.0), (5.0, 50.0), (7.0, 70.0)],
    }
    for instance_id, offset in (("instance_a", 0.0), ("instance_b", 10.0)):
        shifted_real = {
            session_id: (return_value + offset, length_value + offset)
            for session_id, (return_value, length_value) in real_values.items()
        }
        shifted_methods = {
            method: [
                (return_value + offset, length_value + offset)
                for return_value, length_value in values
            ]
            for method, values in methods.items()
        }
        _write_measurement(root, study, instance_id, shifted_real, [10.0 + offset, 20.0 + offset])
        _write_simulations(root, study, instance_id, shifted_methods)

    paths = write_study_outputs(study, root, tmp_path / "analysis")
    rows = read_csv(paths["real_vs_sim"])
    temporal_returns = [
        row
        for row in rows
        if row["latency_method"] == "temporal" and row["metric"] == "episode_return_env"
    ]

    assert len(temporal_returns) == 2
    assert {row["instance_id"] for row in temporal_returns} == {"instance_a", "instance_b"}
    instance_a = next(row for row in temporal_returns if row["instance_id"] == "instance_a")
    assert instance_a["evaluation_scope"] == "heldout"
    assert instance_a["real_session_count"] == "5"
    assert float(instance_a["real_mean"]) == pytest.approx(3.0)
    assert float(instance_a["real_std"]) == pytest.approx(1.58113883)
    assert float(instance_a["sim_mean"]) == pytest.approx(4.0)
    assert float(instance_a["sim_minus_real"]) == pytest.approx(1.0)
    assert instance_a["difference_ci95"] != ""
    assert instance_a["fit_session_ids"] == "0,1,2,3,4"


def test_aggregate_requires_completed_simulation_methods(tmp_path):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    root = tmp_path / "study"
    _write_measurement(root, study, "instance_a", {0: (2.0, 20.0)}, [10.0])
    _write_simulations(
        root,
        study,
        "instance_a",
        {
            "temporal": [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)],
            "iid": [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)],
        },
    )
    (root / "sim/a100/instance_a/iid/DONE").unlink()

    with pytest.raises(FileNotFoundError):
        write_study_outputs(study, root, tmp_path / "analysis")


@pytest.mark.parametrize(
    "relative_marker",
    [
        "measure/a100/instance_a/profile/DONE",
        "measure/a100/instance_a/sessions/0/DONE",
        "sim/a100/instance_a/temporal/0/DONE",
    ],
)
def test_aggregate_requires_leaf_completion_markers(tmp_path, relative_marker):
    study = _study(tmp_path, "cross_device_1real_3sim.yaml")
    root = tmp_path / "study"
    _write_measurement(root, study, "instance_a", {0: (2.0, 20.0)}, [10.0])
    _write_simulations(
        root,
        study,
        "instance_a",
        {
            "temporal": [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)],
            "iid": [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)],
        },
    )
    (root / relative_marker).unlink()

    with pytest.raises(FileNotFoundError):
        write_study_outputs(study, root, tmp_path / "analysis")


def test_real10_constant_sessions_have_zero_difference_interval(tmp_path):
    study = _study(tmp_path, "cross_device_10real_20sim.yaml")
    root = tmp_path / "study"
    real_values = {
        session_id: (2.0, 20.0)
        for session_id in study["sessions"]["evaluation"]
    }
    _write_measurement(root, study, "instance_a", real_values, [10.0])
    _write_simulations(
        root,
        study,
        "instance_a",
        {
            "temporal": [(3.0, 30.0)] * 3,
            "iid": [(3.0, 30.0)] * 3,
        },
    )

    paths = write_study_outputs(study, root, tmp_path / "analysis")
    row = next(
        row
        for row in read_csv(paths["real_vs_sim"])
        if row["latency_method"] == "temporal" and row["metric"] == "episode_return_env"
    )

    assert float(row["sim_minus_real"]) == pytest.approx(1.0)
    assert float(row["difference_ci95"]) == pytest.approx(0.0)


def test_aggregate_requires_a_completed_instance_for_each_device(tmp_path):
    study = load_study("configs/experiments/cross_device_1real_3sim.yaml")
    root = tmp_path / "study"
    _write_measurement(root, study, "instance_a", {0: (2.0, 20.0)}, [10.0])

    with pytest.raises(FileNotFoundError, match="l40s"):
        write_study_outputs(study, root, tmp_path / "analysis")


def test_heldout_aggregate_requires_completed_fit_sessions(tmp_path):
    study = _study(tmp_path, "cross_device_10real_20sim.yaml")
    root = tmp_path / "study"
    real_values = {
        session_id: (2.0, 20.0)
        for session_id in study["sessions"]["evaluation"]
    }
    _write_measurement(root, study, "instance_a", real_values, [10.0])
    (root / "measure/a100/instance_a/sessions/0/DONE").unlink()

    with pytest.raises(FileNotFoundError):
        write_study_outputs(study, root, tmp_path / "analysis")
