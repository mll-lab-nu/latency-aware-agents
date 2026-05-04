from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from latency_bench.analysis.error_metrics import (
    action_age_tv_distance,
    latency_distribution_match_metrics,
    length_error_metrics,
    return_error_metrics,
)
from latency_bench.analysis.eval_stats import (
    ACTION_RECORD_CANDIDATES,
    LATENCY_RECORD_CANDIDATES,
    extract_action_age_frames_from_records,
    extract_latency_samples_from_records,
    load_jsonl_records,
    resolve_run_file,
    summarize_eval_run,
)
from latency_bench.analysis.profile_stats import compute_latency_profile_summary
from latency_bench.latency.profile import load_latency_samples, load_profile_manifest


PROFILE_SUMMARY_COLUMNS = (
    "profile_id",
    "profile_type",
    "latency_scope",
    "policy_id",
    "env_id",
    "target_device",
    "fps",
    "frame_ms",
    "n_warmup",
    "n_samples",
    "batch_size",
    "trace_path",
    "mean_ms",
    "std_ms",
    "min_ms",
    "max_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "mean_effective_frames",
    "p50_frames",
    "p90_frames",
    "p95_frames",
    "p99_frames",
    "prob_latency_gt_1_frame",
    "prob_latency_gt_2_frames",
    "prob_latency_gt_3_frames",
)


def write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV rows")

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _stable_fieldnames(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_latency_profile_summary_rows(profile_paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_path in profile_paths:
        manifest_path, manifest = _load_profile(profile_path)
        trace_path = _resolve_profile_trace_path(manifest_path, manifest)
        samples = load_latency_samples(trace_path)
        summary = compute_latency_profile_summary(samples, float(manifest["fps"]))
        row = {key: manifest.get(key) for key in PROFILE_SUMMARY_COLUMNS}
        row.update(summary)
        row["trace_path"] = manifest.get("trace_path")
        rows.append({key: row.get(key) for key in PROFILE_SUMMARY_COLUMNS})
    return rows


def build_profile_match_rows(match_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in match_specs:
        manifest_path, manifest = _load_profile(spec["real_profile"])
        real_samples = load_latency_samples(_resolve_profile_trace_path(manifest_path, manifest))
        sim_latency_path = resolve_run_file(spec["sim_latency_records"], LATENCY_RECORD_CANDIDATES)
        sim_records = load_jsonl_records(sim_latency_path)
        sim_source = spec.get("sim_latency_source", "auto")
        sim_samples = extract_latency_samples_from_records(sim_records, source=sim_source)
        fps = float(spec.get("fps") or manifest["fps"])

        row: dict[str, Any] = {
            "label": spec.get("label"),
            "real_profile": str(manifest_path),
            "real_profile_id": manifest.get("profile_id"),
            "real_target_device": manifest.get("target_device"),
            "sim_latency_records": str(sim_latency_path),
            "sim_latency_source": sim_source,
            "fps": fps,
        }
        row.update(latency_distribution_match_metrics(real_samples, sim_samples, fps))
        row["action_age_tv"] = _optional_action_age_tv(
            spec.get("real_action_records"),
            spec.get("sim_action_records"),
        )
        rows.append(row)
    return rows


def build_same_device_calibration_rows(calibration_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in calibration_specs:
        real_summary = summarize_eval_run(
            spec["realtime_run"],
            return_field=spec.get("return_field", "episode_return_env"),
            length_field=spec.get("length_field"),
        )
        sim_summary = summarize_eval_run(
            spec["simulated_run"],
            return_field=spec.get("return_field", "episode_return_env"),
            length_field=spec.get("length_field"),
        )

        row: dict[str, Any] = {
            "label": spec.get("label"),
            "policy_id": spec.get("policy_id"),
            "env_id": spec.get("env_id"),
            "profile_id": spec.get("profile_id"),
            "hardware": spec.get("hardware"),
            "realtime_run": spec.get("realtime_run"),
            "simulated_run": spec.get("simulated_run"),
            "realtime_n_episodes": real_summary["n_episodes"],
            "simulated_n_episodes": sim_summary["n_episodes"],
            "realtime_mean_return": real_summary["mean_return"],
            "simulated_mean_return": sim_summary["mean_return"],
            "realtime_mean_length": real_summary["mean_length"],
            "simulated_mean_length": sim_summary["mean_length"],
        }
        row.update(
            _performance_error_fields(
                sim_value=sim_summary["mean_return"],
                real_value=real_summary["mean_return"],
                scale=spec.get("return_scale"),
                allow_fallback=bool(spec.get("allow_realtime_denominator_fallback", False)),
                kind="return",
            )
        )
        row.update(
            _performance_error_fields(
                sim_value=sim_summary["mean_length"],
                real_value=real_summary["mean_length"],
                scale=spec.get("length_scale"),
                allow_fallback=bool(spec.get("allow_realtime_denominator_fallback", False)),
                kind="length",
            )
        )
        rows.append(row)
    return rows


def build_cross_device_prediction_rows(prediction_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in prediction_specs:
        real_summary = summarize_eval_run(
            spec["true_realtime_run"],
            return_field=spec.get("return_field", "episode_return_env"),
            length_field=spec.get("length_field"),
        )
        sim_summary = summarize_eval_run(
            spec["simulated_run"],
            return_field=spec.get("return_field", "episode_return_env"),
            length_field=spec.get("length_field"),
        )
        row: dict[str, Any] = {
            "label": spec.get("label"),
            "policy_id": spec.get("policy_id"),
            "env_id": spec.get("env_id"),
            "target_device": spec.get("target_device"),
            "target_profile": spec.get("target_profile"),
            "replay_host": spec.get("replay_host"),
            "true_realtime_run": spec.get("true_realtime_run"),
            "simulated_run": spec.get("simulated_run"),
            "true_realtime_n_episodes": real_summary["n_episodes"],
            "simulated_n_episodes": sim_summary["n_episodes"],
            "true_realtime_mean_return": real_summary["mean_return"],
            "simulated_mean_return": sim_summary["mean_return"],
            "true_realtime_mean_length": real_summary["mean_length"],
            "simulated_mean_length": sim_summary["mean_length"],
        }
        row.update(
            _performance_error_fields(
                sim_value=sim_summary["mean_return"],
                real_value=real_summary["mean_return"],
                scale=spec.get("return_scale"),
                allow_fallback=bool(spec.get("allow_realtime_denominator_fallback", False)),
                kind="return",
            )
        )
        row.update(
            _performance_error_fields(
                sim_value=sim_summary["mean_length"],
                real_value=real_summary["mean_length"],
                scale=spec.get("length_scale"),
                allow_fallback=bool(spec.get("allow_realtime_denominator_fallback", False)),
                kind="length",
            )
        )
        rows.append(row)
    return rows


def _performance_error_fields(
    *,
    sim_value: float,
    real_value: float,
    scale: float | None,
    allow_fallback: bool,
    kind: str,
) -> dict[str, Any]:
    abs_error = abs(float(sim_value) - float(real_value))
    if scale is None and not allow_fallback:
        return {
            f"abs_{kind}_error": abs_error,
            f"norm_{kind}_error": None,
            f"{kind}_denominator": None,
            f"{kind}_denominator_mode": "missing_scale",
        }

    if kind == "return":
        return return_error_metrics(
            sim_value,
            real_value,
            return_scale=scale,
            fallback_to_realtime_denominator=allow_fallback,
        )
    if kind == "length":
        return length_error_metrics(
            sim_value,
            real_value,
            length_scale=scale,
            fallback_to_realtime_denominator=allow_fallback,
        )
    raise ValueError(f"Unsupported performance error kind: {kind}")


def _optional_action_age_tv(real_action_records: str | Path | None, sim_action_records: str | Path | None) -> float | None:
    if not real_action_records or not sim_action_records:
        return None
    real_path = resolve_run_file(real_action_records, ACTION_RECORD_CANDIDATES)
    sim_path = resolve_run_file(sim_action_records, ACTION_RECORD_CANDIDATES)
    real_ages = extract_action_age_frames_from_records(load_jsonl_records(real_path), strict=True)
    sim_ages = extract_action_age_frames_from_records(load_jsonl_records(sim_path), strict=True)
    return action_age_tv_distance(real_ages, sim_ages)


def _load_profile(profile_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(profile_path)
    manifest_path = path / "profile.json" if path.is_dir() else path
    return manifest_path, load_profile_manifest(manifest_path)


def _resolve_profile_trace_path(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    trace_path = Path(str(manifest["trace_path"]))
    if trace_path.is_absolute():
        return trace_path
    return manifest_path.parent / trace_path


def _stable_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames
