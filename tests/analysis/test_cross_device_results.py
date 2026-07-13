from __future__ import annotations

from pathlib import Path

import pytest

from latency_bench.analysis.cross_device_results import write_cross_device_outputs
from latency_bench.analysis.profile_stats import compute_latency_profile_summary
from latency_bench.cross_device.spec import load_workload
from latency_bench.latency.profile import write_latency_samples
from latency_bench.utils.io import read_csv, write_json, write_jsonl


def _workload() -> dict:
    return load_workload("configs/experiments/cross_device.yaml")


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


def _write_variable_run(path: Path, return_values: list[float]) -> None:
    write_jsonl(
        path / "episode_metrics.jsonl",
        [
            {
                "episode_return_env": value,
                "survival_steps": value * 10.0,
                "metadata": {},
            }
            for value in return_values
        ],
    )
    (path / "DONE").touch()


def _write_raw_session(
    root: Path,
    target: Path,
    session_id: int,
    return_value: float,
    length_value: float,
    samples: list[float],
) -> None:
    session_dir = root / "raw" / target / "sessions" / str(session_id)
    _write_run(session_dir, return_value, length_value)
    write_jsonl(
        session_dir / "e2e_latencies.jsonl",
        [
            {"latency_ms": sample, "bench_overhead_ms": sample / 10.0}
            for sample in samples
        ],
    )


def _write_profile(
    root: Path,
    canonical_ref: Path,
    fit_session_ids: list[int],
    samples: list[float],
) -> None:
    profile_dir = root / "profiles" / canonical_ref
    profile_dir.mkdir(parents=True, exist_ok=True)
    write_latency_samples(samples, profile_dir / "e2e_latencies.jsonl")
    write_latency_samples(
        [sample / 2.0 for sample in samples],
        profile_dir / "infer_latencies.jsonl",
    )
    summary = compute_latency_profile_summary(samples, 20.0)
    inference_summary = compute_latency_profile_summary(
        [sample / 2.0 for sample in samples],
        20.0,
    )
    device_id, instance_id, workload_id, revision = canonical_ref.parts
    write_json(
        profile_dir / "profile.json",
        {
            "profile_id": "L_A100_SXM4_80GB",
            "policy_id": workload_id,
            "env_id": "flappy",
            "device_id": device_id,
            "target_device": device_id,
            "instance_id": instance_id,
            "workload_id": workload_id,
            "profile_revision": revision,
            "fit_session_ids": fit_session_ids,
            "env_fps": 20.0,
            "frame_ms": 50.0,
            "trace_path": "e2e_latencies.jsonl",
            "inference_trace_path": "infer_latencies.jsonl",
            "summary": summary,
            "inference_summary": inference_summary,
        },
    )
    (profile_dir / "DONE").touch()


def _write_simulations(
    root: Path,
    canonical_ref: Path,
    values: dict[str, list[tuple[float, float]]],
) -> None:
    for method, method_values in values.items():
        for seed, (return_value, length_value) in enumerate(method_values):
            _write_run(
                root / "sim" / canonical_ref / method / str(seed),
                return_value,
                length_value,
            )


def _write_complete_selection(
    root: Path,
    canonical_ref: Path,
    fit_session_ids: list[int],
    real_values: dict[int, tuple[float, float]],
) -> None:
    samples = [10.0, 20.0, 30.0]
    for session_id, (return_value, length_value) in real_values.items():
        _write_raw_session(
            root,
            canonical_ref.parent,
            session_id,
            return_value,
            length_value,
            samples,
        )
    _write_profile(root, canonical_ref, fit_session_ids, samples)
    _write_simulations(
        root,
        canonical_ref,
        {
            "temporal": [(2.0, 20.0), (4.0, 40.0), (6.0, 60.0)],
            "iid": [(3.0, 30.0), (5.0, 50.0), (7.0, 70.0)],
        },
    )


def _write_outputs(
    tmp_path: Path,
    root: Path,
    profile_refs: list[Path],
    real_session_ids: list[int],
) -> dict[str, Path]:
    return write_cross_device_outputs(
        _workload(),
        root,
        tmp_path / "analysis",
        profile_refs,
        real_session_ids,
        ["temporal", "iid"],
        [0, 1, 2],
    )


def test_outputs_use_explicit_profile_sessions_and_seeds(tmp_path):
    root = tmp_path / "artifacts"
    canonical_ref = Path("a100-sxm4-80gb/instance_a/openvla-flappy/fit-0")
    _write_complete_selection(root, canonical_ref, [0], {0: (2.0, 20.0)})

    paths = _write_outputs(tmp_path, root, [canonical_ref], [0])

    latency_rows = read_csv(paths["latency_samples"])
    profile_rows = read_csv(paths["latency_profile_summary"])
    result_rows = read_csv(paths["real_vs_sim"])
    assert {row["latency_kind"] for row in latency_rows} == {
        "e2e_latency",
        "inference_latency",
        "overhead_latency",
    }
    assert [
        float(row["latency_ms"])
        for row in latency_rows
        if row["latency_kind"] == "overhead_latency"
    ] == pytest.approx([1.0, 2.0, 3.0])
    assert profile_rows[0]["profile_ref"] == canonical_ref.as_posix()
    assert profile_rows[0]["profile_revision"] == "fit-0"
    assert "study_id" not in profile_rows[0]
    row = next(
        row
        for row in result_rows
        if row["latency_method"] == "temporal"
        and row["metric"] == "episode_return_env"
    )
    assert row["fit_session_ids"] == "0"
    assert row["real_session_ids"] == "0"
    assert row["latency_seed_ids"] == "0,1,2"
    assert row["evaluation_scope"] == "in_sample"
    assert row["real_std"] == ""
    assert row["real_ci95"] == ""
    assert row["difference_ci95"] == ""
    assert float(row["sim_mean"]) == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("fit_ids", "real_ids", "scope"),
    [
        ([0, 1], [0], "in_sample"),
        ([0, 1], [2, 3], "heldout"),
        ([0, 1], [1, 2], "mixed"),
    ],
)
def test_evaluation_scope_comes_from_explicit_selection(
    tmp_path,
    fit_ids,
    real_ids,
    scope,
):
    root = tmp_path / "artifacts"
    canonical_ref = Path("a100-sxm4-80gb/instance_a/openvla-flappy/revision")
    all_ids = set(fit_ids) | set(real_ids)
    _write_complete_selection(
        root,
        canonical_ref,
        fit_ids,
        {session_id: (float(session_id), 20.0) for session_id in all_ids},
    )

    paths = _write_outputs(tmp_path, root, [canonical_ref], real_ids)

    assert {row["evaluation_scope"] for row in read_csv(paths["real_vs_sim"])} == {
        scope
    }


def test_profile_revisions_remain_separate(tmp_path):
    root = tmp_path / "artifacts"
    refs = [
        Path("a100-sxm4-80gb/instance_a/openvla-flappy/fit-0"),
        Path("a100-sxm4-80gb/instance_a/openvla-flappy/fit-0-1"),
    ]
    for canonical_ref in refs:
        _write_complete_selection(root, canonical_ref, [0], {0: (2.0, 20.0)})

    paths = _write_outputs(tmp_path, root, refs, [0])

    assert {row["profile_ref"] for row in read_csv(paths["latency_profile_summary"])} == {
        ref.as_posix() for ref in refs
    }


def test_statistics_use_session_and_seed_means_not_pooled_episodes(tmp_path):
    root = tmp_path / "artifacts"
    canonical_ref = Path("a100-sxm4-80gb/instance_a/openvla-flappy/fit-2")
    _write_raw_session(root, canonical_ref.parent, 2, 2.0, 20.0, [10.0, 20.0])
    _write_profile(root, canonical_ref, [2], [10.0, 20.0])
    _write_variable_run(root / "raw" / canonical_ref.parent / "sessions/0", [0.0])
    _write_variable_run(
        root / "raw" / canonical_ref.parent / "sessions/1",
        [10.0, 10.0, 10.0],
    )
    _write_variable_run(root / "sim" / canonical_ref / "temporal/0", [0.0])
    _write_variable_run(
        root / "sim" / canonical_ref / "temporal/1",
        [20.0, 20.0, 20.0],
    )

    paths = write_cross_device_outputs(
        _workload(),
        root,
        tmp_path / "analysis",
        [canonical_ref],
        [0, 1],
        ["temporal"],
        [0, 1],
    )
    row = next(
        row
        for row in read_csv(paths["real_vs_sim"])
        if row["metric"] == "episode_return_env"
    )

    assert float(row["real_mean"]) == pytest.approx(5.0)
    assert float(row["sim_mean"]) == pytest.approx(10.0)


@pytest.mark.parametrize(
    "relative_marker",
    [
        "profiles/a100-sxm4-80gb/instance_a/openvla-flappy/fit-0/DONE",
        "raw/a100-sxm4-80gb/instance_a/openvla-flappy/sessions/0/DONE",
        "sim/a100-sxm4-80gb/instance_a/openvla-flappy/fit-0/temporal/0/DONE",
    ],
)
def test_selected_leaf_requires_done(tmp_path, relative_marker):
    root = tmp_path / "artifacts"
    canonical_ref = Path("a100-sxm4-80gb/instance_a/openvla-flappy/fit-0")
    _write_complete_selection(root, canonical_ref, [0], {0: (2.0, 20.0)})
    (root / relative_marker).unlink()

    with pytest.raises(FileNotFoundError):
        _write_outputs(tmp_path, root, [canonical_ref], [0])


def test_constant_replicates_have_zero_difference_interval(tmp_path):
    root = tmp_path / "artifacts"
    canonical_ref = Path("a100-sxm4-80gb/instance_a/openvla-flappy/fit-0")
    _write_complete_selection(
        root,
        canonical_ref,
        [0],
        {0: (2.0, 20.0), 1: (2.0, 20.0)},
    )
    _write_simulations(
        root,
        canonical_ref,
        {
            "temporal": [(3.0, 30.0)] * 3,
            "iid": [(3.0, 30.0)] * 3,
        },
    )

    paths = _write_outputs(tmp_path, root, [canonical_ref], [0, 1])
    row = next(
        row
        for row in read_csv(paths["real_vs_sim"])
        if row["latency_method"] == "temporal"
        and row["metric"] == "episode_return_env"
    )

    assert float(row["sim_minus_real"]) == pytest.approx(1.0)
    assert float(row["difference_ci95"]) == pytest.approx(0.0)
