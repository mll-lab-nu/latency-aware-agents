from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

from latency_bench.analysis.eval_stats import summarize_eval_run
from latency_bench.latency.profile import load_profile_manifest


EXPERIMENT1_COLUMNS = (
    "experiment_id",
    "panel",
    "policy",
    "env",
    "profile_id",
    "hardware",
    "env_fps",
    "mean_ms",
    "std_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "mean_effective_frames",
    "p95_frames",
    "n_samples",
    "notes",
)

EXPERIMENT2_COLUMNS = (
    "experiment_id",
    "panel",
    "policy",
    "env",
    "profile_id",
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
    "replay_device",
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
            profile_path = _profile_path_for_row(root_path, row_spec, device)
            manifest = _latest_manifest(
                manifests,
                stage="measure",
                experiment_id=experiment_id,
                row=row_spec,
                target_device_id=device["id"],
            )
            if manifest and manifest.get("profile_path"):
                profile_path = Path(str(manifest["profile_path"]))
            summary, profile_metadata, notes = _load_profile_summary(profile_path)
            notes = _join_notes(
                notes,
                _skipped_notes_if_missing(
                    manifests,
                    matches=[manifest] if manifest and not notes else [],
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
            realtime = _matching_manifests(
                manifests,
                stage="realtime",
                experiment_id=experiment_id,
                row=row_spec,
                target_device_id=device["id"],
            )
            sim = _matching_manifests(
                manifests,
                stage="sim_same",
                experiment_id=experiment_id,
                row=row_spec,
                target_device_id=device["id"],
            )
            real_summary = _summarize_manifest_runs(realtime)
            sim_summary = _summarize_manifest_runs(sim)
            notes = _join_notes(
                _missing_note("realtime", realtime, real_summary),
                _skipped_notes_if_missing(
                    manifests,
                    matches=realtime,
                    stage="realtime",
                    experiment_id=experiment_id,
                    row=row_spec,
                    target_device_id=device["id"],
                ),
                _missing_note("sim-same", sim, sim_summary),
                _skipped_notes_if_missing(
                    manifests,
                    matches=sim,
                    stage="sim_same",
                    experiment_id=experiment_id,
                    row=row_spec,
                    target_device_id=device["id"],
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
            true_realtime = _matching_manifests(
                manifests,
                stage="realtime",
                experiment_id=experiment_id,
                row=row_spec,
                target_device_id=target["id"],
            )
            true_summary = _summarize_manifest_runs(true_realtime)
            sim_cross = _matching_manifests(
                manifests,
                stage="sim_cross",
                experiment_id=experiment_id,
                row=row_spec,
                target_device_id=target["id"],
                replay_device_id=replay["id"],
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
                _skipped_notes_if_missing(
                    manifests,
                    matches=true_realtime,
                    stage="realtime",
                    experiment_id=experiment_id,
                    row=row_spec,
                    target_device_id=target["id"],
                ),
                _missing_note("sim-cross", sim_cross, sim_summary),
                _skipped_notes_if_missing(
                    manifests,
                    matches=sim_cross,
                    stage="sim_cross",
                    experiment_id=experiment_id,
                    row=row_spec,
                    target_device_id=target["id"],
                    replay_device_id=replay["id"],
                ),
            )
            out = {
                "experiment_id": experiment_id,
                "panel": experiment.get("panel"),
                "policy": row_spec.get("policy"),
                "env": row_spec.get("env"),
                "target_device": target.get("display_name", target.get("id")),
                "target_profile": target.get("profile_id"),
                "replay_device": replay.get("display_name", replay.get("id")),
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


def load_protocol_manifests(root: str | Path, *, raw_root: str | Path | None = None) -> list[dict[str, Any]]:
    if raw_root is not None:
        return load_raw_run_manifests(raw_root)
    return load_run_manifests(root)


def load_run_manifests(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / "manifests" / "runs.jsonl"
    if not path.exists():
        return []
    records = _load_manifest_jsonl(path)
    return _dedupe_manifest_records(records)


def load_raw_run_manifests(raw_root: str | Path) -> list[dict[str, Any]]:
    root = Path(raw_root)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    manifest_paths = sorted(root.glob("**/cross_device/manifests/runs.jsonl"))
    for manifest_path in manifest_paths:
        protocol_root = manifest_path.parents[1]
        for record in _load_manifest_jsonl(manifest_path):
            records.append(_remap_manifest_paths(record, protocol_root))
    return _dedupe_manifest_records(records)


def _load_manifest_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Manifest record must be a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def _remap_manifest_paths(record: dict[str, Any], protocol_root: Path) -> dict[str, Any]:
    remapped = dict(record)
    for key in ("output_dir", "profile_path", "generated_config_path"):
        value = remapped.get(key)
        if not value:
            continue
        remapped[key] = _remap_protocol_path(str(value), protocol_root)
    return remapped


def _remap_protocol_path(value: str, protocol_root: Path) -> str:
    path = Path(value)
    parts = path.parts
    suffix = _protocol_path_suffix(parts)
    if not suffix:
        return value
    return str(protocol_root.joinpath(*suffix))


def _protocol_path_suffix(parts: tuple[str, ...]) -> tuple[str, ...]:
    for marker in ("cross_device",):
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == marker:
                return tuple(parts[index + 1 :])
    for marker in ("runs", "profiles", "generated_configs", "manifests"):
        if marker in parts:
            index = parts.index(marker)
            return tuple(parts[index:])
    return ()


def _dedupe_manifest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed_by_run_key: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}
    non_completed_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for record in records:
        output_dir = record.get("output_dir")
        if not output_dir:
            passthrough.append(record)
            continue
        run_key = (
            str(record.get("stage")),
            str(record.get("experiment_id")),
            str(record.get("policy")),
            str(record.get("env")),
            str(record.get("target_device_id")),
            str(record.get("replay_device_id")),
            str(record.get("seed")),
        )
        if record.get("status") == "completed":
            existing = completed_by_run_key.get(run_key)
            if existing is None or _record_sort_timestamp(record) >= _record_sort_timestamp(existing):
                completed_by_run_key[run_key] = record
            continue
        skipped_key = (
            *run_key,
            str(record.get("status")),
            str(record.get("skip_reason") or record.get("notes") or ""),
            str(record.get("runtime_env_detected_raw") or ""),
        )
        non_completed_by_key[skipped_key] = record
    return [*passthrough, *non_completed_by_key.values(), *completed_by_run_key.values()]


def _record_sort_timestamp(record: dict[str, Any]) -> str:
    return str(record.get("completed_at") or record.get("started_at") or "")


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


def _profile_path_for_row(root: Path, row: dict[str, Any], device: dict[str, Any]) -> Path:
    return root / "profiles" / slugify(row["policy"]) / slugify(row["env"]) / str(device["profile_id"]) / "profile.json"


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "unknown"


def _load_profile_summary(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"missing profile: {path}")
    manifest = load_profile_manifest(path)
    summary = dict(manifest["summary"])
    summary["n_samples"] = manifest["n_samples"]
    return summary, manifest, ""


def _matching_manifests(
    manifests: list[dict[str, Any]],
    *,
    stage: str,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
    replay_device_id: str | None = None,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for manifest in manifests:
        if manifest.get("stage") != stage:
            continue
        if not _manifest_contributes_to(manifest, experiment_id):
            continue
        if manifest.get("policy") != row.get("policy") or manifest.get("env") != row.get("env"):
            continue
        if manifest.get("target_device_id") != target_device_id:
            continue
        if replay_device_id is not None and manifest.get("replay_device_id") != replay_device_id:
            continue
        if manifest.get("status") != "completed":
            continue
        matches.append(manifest)
    return sorted(matches, key=lambda item: str(item.get("completed_at") or item.get("started_at") or ""))


def _skipped_notes_if_missing(
    manifests: list[dict[str, Any]],
    *,
    matches: list[dict[str, Any]],
    stage: str,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
    replay_device_id: str | None = None,
) -> str:
    if matches:
        return ""
    return _matching_skipped_notes(
        manifests,
        stage=stage,
        experiment_id=experiment_id,
        row=row,
        target_device_id=target_device_id,
        replay_device_id=replay_device_id,
    )


def _matching_skipped_notes(
    manifests: list[dict[str, Any]],
    *,
    stage: str,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
    replay_device_id: str | None = None,
) -> str:
    notes: list[str] = []
    for manifest in manifests:
        if manifest.get("stage") != stage:
            continue
        if not _manifest_contributes_to(manifest, experiment_id):
            continue
        if manifest.get("policy") != row.get("policy") or manifest.get("env") != row.get("env"):
            continue
        if manifest.get("target_device_id") != target_device_id:
            continue
        if replay_device_id is not None and manifest.get("replay_device_id") != replay_device_id:
            continue
        if manifest.get("status") not in {"skipped", "failed"}:
            continue
        note = manifest.get("skip_reason") or manifest.get("notes")
        if note:
            notes.append(str(note))
    return "; ".join(dict.fromkeys(notes))


def _manifest_contributes_to(manifest: dict[str, Any], experiment_id: str) -> bool:
    experiment_ids = manifest.get("experiment_ids")
    if isinstance(experiment_ids, list):
        return experiment_id in {str(item) for item in experiment_ids}
    return manifest.get("experiment_id") == experiment_id


def _latest_manifest(
    manifests: list[dict[str, Any]],
    *,
    stage: str,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
) -> dict[str, Any] | None:
    matches = _matching_manifests(
        manifests,
        stage=stage,
        experiment_id=experiment_id,
        row=row,
        target_device_id=target_device_id,
    )
    return matches[-1] if matches else None


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
