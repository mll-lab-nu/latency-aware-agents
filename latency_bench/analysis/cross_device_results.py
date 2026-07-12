from __future__ import annotations

from pathlib import Path
from typing import Any

from latency_bench.analysis.eval_stats import (
    independent_mean_difference_ci95,
    summarize_eval_sessions,
)
from latency_bench.cross_device.artifacts import raw_session_dir
from latency_bench.latency.profile import load_latency_samples
from latency_bench.utils.io import load_jsonl_records, read_json, write_csv


PROFILE_SUMMARY_FIELDS = (
    "n_samples",
    "mean_ms",
    "std_ms",
    "min_ms",
    "max_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "frame_ms",
    "mean_effective_frames",
    "p50_frames",
    "p90_frames",
    "p95_frames",
    "p99_frames",
    "prob_latency_gt_1_frame",
    "prob_latency_gt_2_frames",
    "prob_latency_gt_3_frames",
)


def write_cross_device_outputs(
    workload: dict[str, Any],
    root: str | Path,
    output_dir: str | Path,
    profile_refs: list[Path],
    real_session_ids: list[int],
    methods: list[str],
    latency_seeds: list[int],
) -> dict[str, Path]:
    root = Path(root)
    output_dir = Path(output_dir)
    devices = {device["id"]: device for device in workload["devices"]}
    latency_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    for canonical_profile_ref in profile_refs:
        profile_dir = root / "profiles" / canonical_profile_ref
        (profile_dir / "DONE").read_bytes()
        profile = read_json(profile_dir / "profile.json")
        identity = _identity(
            devices[profile["device_id"]],
            canonical_profile_ref,
            profile,
        )
        e2e_samples = load_latency_samples(profile_dir / profile["trace_path"])
        inference_samples = load_latency_samples(
            profile_dir / profile["inference_trace_path"]
        )
        target = canonical_profile_ref.parent
        overhead_samples = []
        for session_id in profile["fit_session_ids"]:
            session_dir = raw_session_dir(root, target, session_id)
            (session_dir / "DONE").read_bytes()
            overhead_samples.extend(
                record["bench_overhead_ms"]
                for record in load_jsonl_records(session_dir / "e2e_latencies.jsonl")
            )
        for latency_kind, samples in (
            ("e2e_latency", e2e_samples),
            ("inference_latency", inference_samples),
            ("overhead_latency", overhead_samples),
        ):
            latency_rows.extend(
                {
                    **identity,
                    "sample_index": sample_index,
                    "latency_ms": latency_ms,
                    "env_fps": profile["env_fps"],
                    "frame_ms": profile["frame_ms"],
                    "latency_kind": latency_kind,
                }
                for sample_index, latency_ms in enumerate(samples)
            )
        profile_rows.append(
            {
                **identity,
                "env_fps": profile["env_fps"],
                "inference_mean_ms": profile["inference_summary"]["mean_ms"],
                "e2e_mean_ms": profile["summary"]["mean_ms"],
                "e2e_mean_frames": profile["summary"]["mean_effective_frames"],
                **{
                    field: profile["summary"][field]
                    for field in PROFILE_SUMMARY_FIELDS
                },
            }
        )
        evaluation_rows.extend(
            _evaluation_rows(
                root,
                canonical_profile_ref,
                profile,
                identity,
                real_session_ids,
                methods,
                latency_seeds,
            )
        )

    paths = {
        "latency_samples": output_dir / "latency_samples.csv",
        "latency_profile_summary": output_dir / "latency_profile_summary.csv",
        "real_vs_sim": output_dir / "real_vs_sim.csv",
    }
    write_csv(paths["latency_samples"], latency_rows)
    write_csv(paths["latency_profile_summary"], profile_rows)
    write_csv(paths["real_vs_sim"], evaluation_rows)
    return paths


def _identity(
    device: dict[str, Any],
    canonical_profile_ref: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    return {
        "workload_id": profile["workload_id"],
        "policy_id": profile["policy_id"],
        "env_id": profile["env_id"],
        "device_id": profile["device_id"],
        "device_name": device["display_name"],
        "instance_id": profile["instance_id"],
        "profile_id": profile["profile_id"],
        "profile_revision": profile["profile_revision"],
        "profile_ref": canonical_profile_ref.as_posix(),
        "fit_session_ids": ",".join(
            str(value) for value in profile["fit_session_ids"]
        ),
    }


def _evaluation_rows(
    root: Path,
    canonical_profile_ref: Path,
    profile: dict[str, Any],
    identity: dict[str, Any],
    real_session_ids: list[int],
    methods: list[str],
    latency_seeds: list[int],
) -> list[dict[str, Any]]:
    target = canonical_profile_ref.parent
    real_paths = []
    for session_id in real_session_ids:
        session_dir = raw_session_dir(root, target, session_id)
        (session_dir / "DONE").read_bytes()
        real_paths.append(session_dir)
    real = summarize_eval_sessions(real_paths)
    evaluation_scope = _evaluation_scope(
        profile["fit_session_ids"],
        real_session_ids,
    )
    real_ids = ",".join(str(value) for value in real_session_ids)
    seed_ids = ",".join(str(value) for value in latency_seeds)
    rows = []
    for method in methods:
        sim_paths = []
        for latency_seed in latency_seeds:
            run_dir = (
                root
                / "sim"
                / canonical_profile_ref
                / method
                / str(latency_seed)
            )
            (run_dir / "DONE").read_bytes()
            sim_paths.append(run_dir)
        sim = summarize_eval_sessions(sim_paths)
        selection = {
            **identity,
            "real_session_ids": real_ids,
            "latency_seed_ids": seed_ids,
        }
        rows.extend(
            [
                _metric_row(
                    selection,
                    method,
                    evaluation_scope,
                    "episode_return_env",
                    real,
                    sim,
                    "return",
                ),
                _metric_row(
                    selection,
                    method,
                    evaluation_scope,
                    "survival_steps",
                    real,
                    sim,
                    "length",
                ),
            ]
        )
    return rows


def _evaluation_scope(fit_session_ids: list[int], real_session_ids: list[int]) -> str:
    fit_ids = set(fit_session_ids)
    real_ids = set(real_session_ids)
    if real_ids <= fit_ids:
        return "in_sample"
    if real_ids.isdisjoint(fit_ids):
        return "heldout"
    return "mixed"


def _metric_row(
    identity: dict[str, Any],
    method: str,
    evaluation_scope: str,
    metric: str,
    real: dict[str, Any],
    sim: dict[str, Any],
    summary_key: str,
) -> dict[str, Any]:
    real_mean = real[f"mean_{summary_key}"]
    real_std = real[f"std_{summary_key}"]
    sim_mean = sim[f"mean_{summary_key}"]
    sim_std = sim[f"std_{summary_key}"]
    difference = sim_mean - real_mean
    difference_ci95 = None
    if real["session_count"] > 1 and sim["session_count"] > 1:
        difference, difference_ci95 = independent_mean_difference_ci95(
            first_mean=real_mean,
            first_std=real_std,
            first_count=real["session_count"],
            second_mean=sim_mean,
            second_std=sim_std,
            second_count=sim["session_count"],
        )
    return {
        **identity,
        "latency_method": method,
        "metric": metric,
        "evaluation_scope": evaluation_scope,
        "real_session_count": real["session_count"],
        "real_episode_count": real["n_episodes"],
        "real_mean": real_mean,
        "real_std": real_std,
        "real_ci95": real[f"{summary_key}_ci95"],
        "sim_seed_count": sim["session_count"],
        "sim_episode_count": sim["n_episodes"],
        "sim_mean": sim_mean,
        "sim_std": sim_std,
        "sim_ci95": sim[f"{summary_key}_ci95"],
        "sim_minus_real": difference,
        "difference_ci95": difference_ci95,
    }
