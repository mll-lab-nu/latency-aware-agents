from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from latency_bench.analysis.eval_stats import summarize_eval_run
from latency_bench.latency.profile import load_profile_manifest
from latency_bench.cross_device.layout import slugify
from latency_bench.cross_device.manifest import (
    load_protocol_manifests,
    matching_manifests,
    profile_instances_for_cross_pair,
    profile_instances_for_device,
    replay_host_instances_for_cross_pair,
    skipped_notes_if_missing,
)


EXPERIMENT1_COLUMNS = (
    "experiment_id",
    "panel",
    "policy",
    "env",
    "profile_id",
    "profile_instance_id",
    "hardware",
    "env_fps",
    "inference_mean_ms",
    "e2e_mean_ms",
    "e2e_mean_frames",
    "n_samples",
    "notes",
)

EXPERIMENT2_COLUMNS = (
    "experiment_id",
    "panel",
    "policy",
    "env",
    "profile_id",
    "profile_instance_id",
    "hardware",
    "realtime_return",
    "simulated_return_using_L_d",
    "return_abs_error",
    "return_rel_error_pct",
    "realtime_length",
    "sim_length",
    "length_abs_error",
    "length_rel_error_pct",
    "n_episodes",
    "seed_set",
    "notes",
)

EXPERIMENT3_COLUMNS = (
    "experiment_id",
    "panel",
    "policy",
    "env",
    "target_device",
    "target_profile",
    "target_profile_instance_id",
    "replay_device",
    "replay_host_instance_id",
    "true_realtime_return_on_d",
    "simulated_return_on_replay_using_L_d",
    "return_abs_error",
    "return_rel_error_pct",
    "true_realtime_length_on_d",
    "simulated_length_on_replay_using_L_d",
    "length_abs_error",
    "length_rel_error_pct",
    "n_episodes",
    "seed_set",
    "use_in_main_claim",
    "notes",
)


def load_protocol_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    with spec_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"Protocol spec must be a mapping: {spec_path}")
    return spec


def build_experiment1_rows(
    spec: dict[str, Any],
    root: str | Path,
    *,
    raw_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root_path = Path(root)
    manifests = load_protocol_manifests(root_path, raw_root=raw_root)
    rows: list[dict[str, Any]] = []
    for experiment_id, experiment, row_spec in _iter_rows(spec, "exp1_latency_profile_summary"):
        for device in _devices(spec):
            matches = matching_manifests(
                manifests,
                stage="measure",
                experiment_id=experiment_id,
                row=row_spec,
                target_device_id=device["id"],
            )
            if not matches:
                notes = _join_notes(
                    _missing_note("profile", matches, {}),
                    skipped_notes_if_missing(
                        manifests,
                        matches=matches,
                        stage="measure",
                        experiment_id=experiment_id,
                        row=row_spec,
                        target_device_id=device["id"],
                    ),
                )
                out = {
                    "experiment_id": experiment_id,
                    "panel": experiment.get("panel"),
                    "policy": row_spec.get("policy"),
                    "env": row_spec.get("env"),
                    "profile_id": device.get("profile_id"),
                    "profile_instance_id": "",
                    "hardware": device.get("display_name", device.get("id")),
                    "notes": notes,
                }
                rows.append({key: out.get(key, "") for key in EXPERIMENT1_COLUMNS})
                continue
            latest_by_instance = _latest_by_profile_instance(matches)
            for profile_instance_id, manifest in latest_by_instance.items():
                profile_path = Path(str(manifest["profile_path"]))
                summary, profile_metadata, notes = _load_profile_summary(profile_path)
                out = {
                    "experiment_id": experiment_id,
                    "panel": experiment.get("panel"),
                    "policy": row_spec.get("policy"),
                    "env": row_spec.get("env"),
                    "profile_id": device.get("profile_id"),
                    "profile_instance_id": profile_instance_id,
                    "hardware": device.get("display_name", device.get("id")),
                    "env_fps": profile_metadata["env_fps"],
                    "notes": notes,
                }
                out.update({key: summary.get(key, "") for key in EXPERIMENT1_COLUMNS if key not in out})
                rows.append({key: out.get(key, "") for key in EXPERIMENT1_COLUMNS})
    return rows


def build_experiment2_rows(
    spec: dict[str, Any],
    root: str | Path,
    *,
    raw_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root_path = Path(root)
    manifests = load_protocol_manifests(root_path, raw_root=raw_root)
    rows: list[dict[str, Any]] = []
    seed_label = _seed_set_label(spec.get("seed_set", []))
    for experiment_id, experiment, row_spec in _iter_rows(spec, "exp2_same_device_calibration"):
        for device in _devices(spec):
            profile_instances = profile_instances_for_device(
                manifests,
                experiment_id=experiment_id,
                row=row_spec,
                target_device_id=device["id"],
                stages=("measure", "sim_same"),
            )
            if not profile_instances:
                profile_instances = [None]
            for profile_instance_id in profile_instances:
                realtime = matching_manifests(
                    manifests,
                    stage="measure",
                    experiment_id=experiment_id,
                    row=row_spec,
                    target_device_id=device["id"],
                    target_profile_instance_id=profile_instance_id,
                )
                sim = matching_manifests(
                    manifests,
                    stage="sim_same",
                    experiment_id=experiment_id,
                    row=row_spec,
                    target_device_id=device["id"],
                    target_profile_instance_id=profile_instance_id,
                )
                real_summary = _summarize_manifest_runs(realtime)
                sim_summary = _summarize_manifest_runs(sim)
                notes = _join_notes(
                    _missing_note("realtime", realtime, real_summary),
                    skipped_notes_if_missing(
                        manifests,
                        matches=realtime,
                        stage="measure",
                        experiment_id=experiment_id,
                        row=row_spec,
                        target_device_id=device["id"],
                        target_profile_instance_id=profile_instance_id,
                    ),
                    _missing_note("sim-same", sim, sim_summary),
                    skipped_notes_if_missing(
                        manifests,
                        matches=sim,
                        stage="sim_same",
                        experiment_id=experiment_id,
                        row=row_spec,
                        target_device_id=device["id"],
                        target_profile_instance_id=profile_instance_id,
                    ),
                )
                errors = _error_fields(
                    real_value=_maybe_number(real_summary.get("mean_return")),
                    sim_value=_maybe_number(sim_summary.get("mean_return")),
                    real_length=_maybe_number(real_summary.get("mean_length")),
                    sim_length=_maybe_number(sim_summary.get("mean_length")),
                )
                n_episodes = _available_n_episodes(real_summary, sim_summary)
                out = {
                    "experiment_id": experiment_id,
                    "panel": experiment.get("panel"),
                    "policy": row_spec.get("policy"),
                    "env": row_spec.get("env"),
                    "profile_id": device.get("profile_id"),
                    "profile_instance_id": profile_instance_id or "",
                    "hardware": device.get("display_name", device.get("id")),
                    "realtime_return": real_summary.get("mean_return", ""),
                    "simulated_return_using_L_d": sim_summary.get("mean_return", ""),
                    "return_abs_error": errors["return_abs_error"],
                    "return_rel_error_pct": errors["return_rel_error_pct"],
                    "realtime_length": real_summary.get("mean_length", ""),
                    "sim_length": sim_summary.get("mean_length", ""),
                    "length_abs_error": errors["length_abs_error"],
                    "length_rel_error_pct": errors["length_rel_error_pct"],
                    "n_episodes": n_episodes,
                    "seed_set": seed_label,
                    "notes": notes,
                }
                rows.append({key: out.get(key, "") for key in EXPERIMENT2_COLUMNS})
    return rows


def build_experiment3_rows(
    spec: dict[str, Any],
    root: str | Path,
    *,
    raw_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root_path = Path(root)
    manifests = load_protocol_manifests(root_path, raw_root=raw_root)
    rows: list[dict[str, Any]] = []
    seed_label = _seed_set_label(spec.get("seed_set", []))
    for experiment_id, experiment, row_spec in _iter_rows(spec, "exp3_cross_device_prediction"):
        devices_by_id = _devices_by_id(spec)
        for pair in experiment["replay_pairs"]:
            target = devices_by_id[str(pair["target_device"])]
            replay = devices_by_id[str(pair["replay_device"])]
            target_instances = profile_instances_for_cross_pair(
                manifests,
                experiment_id=experiment_id,
                row=row_spec,
                target_device_id=target["id"],
                replay_device_id=replay["id"],
            )
            if not target_instances:
                target_instances = [None]
            for target_profile_instance_id in target_instances:
                replay_host_instances = replay_host_instances_for_cross_pair(
                    manifests,
                    experiment_id=experiment_id,
                    row=row_spec,
                    target_device_id=target["id"],
                    target_profile_instance_id=target_profile_instance_id,
                    replay_device_id=replay["id"],
                )
                if not replay_host_instances:
                    replay_host_instances = [None]
                for replay_host_instance_id in replay_host_instances:
                    true_realtime = matching_manifests(
                        manifests,
                        stage="measure",
                        experiment_id=experiment_id,
                        row=row_spec,
                        target_device_id=target["id"],
                        target_profile_instance_id=target_profile_instance_id,
                    )
                    true_summary = _summarize_manifest_runs(true_realtime)
                    sim_cross = matching_manifests(
                        manifests,
                        stage="sim_cross",
                        experiment_id=experiment_id,
                        row=row_spec,
                        target_device_id=target["id"],
                        target_profile_instance_id=target_profile_instance_id,
                        replay_device_id=replay["id"],
                        replay_host_instance_id=replay_host_instance_id,
                    )
                    sim_summary = _summarize_manifest_runs(sim_cross)
                    errors = _error_fields(
                        real_value=_maybe_number(true_summary.get("mean_return")),
                        sim_value=_maybe_number(sim_summary.get("mean_return")),
                        real_length=_maybe_number(true_summary.get("mean_length")),
                        sim_length=_maybe_number(sim_summary.get("mean_length")),
                    )
                    notes = _join_notes(
                        _missing_note("true realtime", true_realtime, true_summary),
                        skipped_notes_if_missing(
                            manifests,
                            matches=true_realtime,
                            stage="measure",
                            experiment_id=experiment_id,
                            row=row_spec,
                            target_device_id=target["id"],
                            target_profile_instance_id=target_profile_instance_id,
                        ),
                        _missing_note("sim-cross", sim_cross, sim_summary),
                        skipped_notes_if_missing(
                            manifests,
                            matches=sim_cross,
                            stage="sim_cross",
                            experiment_id=experiment_id,
                            row=row_spec,
                            target_device_id=target["id"],
                            target_profile_instance_id=target_profile_instance_id,
                            replay_device_id=replay["id"],
                            replay_host_instance_id=replay_host_instance_id,
                        ),
                    )
                    out = {
                        "experiment_id": experiment_id,
                        "panel": experiment.get("panel"),
                        "policy": row_spec.get("policy"),
                        "env": row_spec.get("env"),
                        "target_device": target.get("display_name", target.get("id")),
                        "target_profile": target.get("profile_id"),
                        "target_profile_instance_id": target_profile_instance_id or "",
                        "replay_device": replay.get("display_name", replay.get("id")),
                        "replay_host_instance_id": replay_host_instance_id or "",
                        "true_realtime_return_on_d": true_summary.get("mean_return", ""),
                        "simulated_return_on_replay_using_L_d": sim_summary.get("mean_return", ""),
                        "return_abs_error": errors["return_abs_error"],
                        "return_rel_error_pct": errors["return_rel_error_pct"],
                        "true_realtime_length_on_d": true_summary.get("mean_length", ""),
                        "simulated_length_on_replay_using_L_d": sim_summary.get("mean_length", ""),
                        "length_abs_error": errors["length_abs_error"],
                        "length_rel_error_pct": errors["length_rel_error_pct"],
                        "n_episodes": _available_n_episodes(true_summary, sim_summary),
                        "seed_set": seed_label,
                        "use_in_main_claim": "TRUE",
                        "notes": notes,
                    }
                    rows.append({key: out.get(key, "") for key in EXPERIMENT3_COLUMNS})
    return rows


def write_protocol_tables(
    *,
    spec: dict[str, Any],
    root: str | Path,
    out: str | Path,
    overwrite: bool = False,
    raw_root: str | Path | None = None,
) -> dict[str, Path]:
    out_path = Path(out)
    outputs = {
        "experiment1": out_path / "experiment1_latency_profile_summary.csv",
        "experiment2": out_path / "experiment2_same_device_calibration.csv",
        "experiment3": out_path / "experiment3_cross_device_prediction.csv",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing table files without --overwrite: {existing}")
    out_path.mkdir(parents=True, exist_ok=True)
    _write_csv(outputs["experiment1"], EXPERIMENT1_COLUMNS, build_experiment1_rows(spec, root, raw_root=raw_root))
    _write_csv(outputs["experiment2"], EXPERIMENT2_COLUMNS, build_experiment2_rows(spec, root, raw_root=raw_root))
    _write_csv(outputs["experiment3"], EXPERIMENT3_COLUMNS, build_experiment3_rows(spec, root, raw_root=raw_root))
    return outputs


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _iter_rows(
    spec: dict[str, Any],
    experiment_id_filter: str,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    experiment = spec["experiments"][experiment_id_filter]
    rows = experiment["rows"]
    return [
        (experiment_id_filter, experiment, row)
        for row in rows
    ]


def _devices(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return list(spec["devices"])


def _devices_by_id(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(device["id"]): device for device in _devices(spec)}


def _load_profile_summary(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"missing profile: {path}")
    manifest = load_profile_manifest(path)
    e2e_summary = manifest["e2e_summary"]
    inference_summary = manifest["inference_summary"]
    summary = {
        "inference_mean_ms": inference_summary["mean_ms"],
        "e2e_mean_ms": e2e_summary["mean_ms"],
        "e2e_mean_frames": e2e_summary["mean_effective_frames"],
        "n_samples": manifest["n_samples"],
    }
    return summary, manifest, ""


def _latest_by_profile_instance(manifests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        latest[str(manifest["target_profile_instance_id"])] = manifest
    return dict(sorted(latest.items()))


def _summarize_manifest_runs(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    notes: list[str] = []
    for manifest in manifests:
        output_dir = manifest.get("output_dir")
        if not output_dir:
            notes.append("manifest missing output_dir")
            continue
        summaries.append(summarize_eval_run(output_dir))
    if not summaries:
        return {"notes": "; ".join(notes)}
    total_n = sum(int(summary["n_episodes"]) for summary in summaries)
    if total_n <= 0:
        return {"notes": "; ".join(notes)}
    return {
        "n_episodes": total_n,
        "mean_return": sum(float(summary["mean_return"]) * int(summary["n_episodes"]) for summary in summaries) / total_n,
        "mean_length": sum(float(summary["mean_length"]) * int(summary["n_episodes"]) for summary in summaries) / total_n,
        "notes": "; ".join(notes),
    }


def _error_fields(
    *,
    real_value: float | None,
    sim_value: float | None,
    real_length: float | None,
    sim_length: float | None,
) -> dict[str, Any]:
    return {
        "return_abs_error": _abs_error(sim_value, real_value),
        "return_rel_error_pct": _rel_error_pct(sim_value, real_value),
        "length_abs_error": _abs_error(sim_length, real_length),
        "length_rel_error_pct": _rel_error_pct(sim_length, real_length),
    }


def _abs_error(sim_value: float | None, real_value: float | None) -> float | str:
    if sim_value is None or real_value is None:
        return ""
    return abs(sim_value - real_value)


def _rel_error_pct(sim_value: float | None, real_value: float | None) -> float | str:
    if sim_value is None or real_value is None:
        return ""
    return 100.0 * abs(sim_value - real_value) / max(abs(real_value), 1e-8)


def _maybe_number(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def _available_n_episodes(*summaries: dict[str, Any]) -> int | str:
    values = [summary.get("n_episodes") for summary in summaries if summary.get("n_episodes") not in (None, "")]
    if not values:
        return ""
    return int(min(values))


def _seed_set_label(seed_set: Any) -> str:
    if seed_set is None:
        return ""
    if isinstance(seed_set, (list, tuple)):
        return ",".join(str(item) for item in seed_set)
    return str(seed_set)


def _missing_note(label: str, manifests: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if not manifests:
        return f"missing {label} run"
    if summary.get("notes"):
        return str(summary["notes"])
    return ""


def _join_notes(*notes: str) -> str:
    return "; ".join(note for note in notes if note)
