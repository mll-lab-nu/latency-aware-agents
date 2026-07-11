from __future__ import annotations

from pathlib import Path
from typing import Any

from latency_bench.analysis.eval_stats import (
    independent_mean_difference_ci95,
    summarize_eval_run,
    summarize_eval_sessions,
)
from latency_bench.latency.profile import load_profile_manifest
from latency_bench.cross_device.manifest import (
    load_run_manifests,
    matching_manifests,
    matching_skipped_notes,
    profile_instances_for_cross_pair,
    profile_instances_for_device,
    replay_host_instances_for_cross_pair,
)
from latency_bench.utils.io import read_json, write_csv


EXPERIMENT1_COLUMNS = (
    "experiment_id",
    "panel",
    "policy",
    "env",
    "profile_id",
    "profile_instance_id",
    "profile_build_id",
    "profile_path",
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
    "profile_build_id",
    "latency_method",
    "hardware",
    "sim_host_device_id",
    "sim_host_instance_id",
    "realtime_return",
    "simulated_return_using_L_d",
    "return_abs_error",
    "return_rel_error_pct",
    "realtime_length",
    "sim_length",
    "length_abs_error",
    "length_rel_error_pct",
    "realtime_session_count",
    "simulated_session_count",
    "realtime_return_std",
    "simulated_return_std",
    "realtime_return_ci95",
    "simulated_return_ci95",
    "sim_minus_real_return",
    "sim_minus_real_ci95",
    "realtime_length_std",
    "simulated_length_std",
    "realtime_length_ci95",
    "simulated_length_ci95",
    "realtime_n_episodes",
    "simulated_n_episodes",
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


def _completion_records(
    spec: dict[str, Any],
    root: Path,
    raw_root: str | Path | None,
    stage: str,
) -> list[tuple[dict[str, Any], Path]]:
    paths_and_roots: list[tuple[Path, Path]] = []
    if raw_root is not None:
        for path in Path(raw_root).glob(
            f"**/cross_device/{spec['protocol_id']}/completion/{stage}/**/*.json"
        ):
            protocol_root = next(
                parent for parent in path.parents if parent.name == spec["protocol_id"]
            )
            paths_and_roots.append((path, protocol_root))
    paths_and_roots.extend(
        (path, root) for path in (root / "completion" / stage).glob("**/*.json")
    )

    records: dict[tuple[Any, ...], tuple[dict[str, Any], Path]] = {}
    for path, protocol_root in paths_and_roots:
        completion = read_json(path)
        if completion["protocol_id"] != spec["protocol_id"]:
            continue
        key = (
            completion["target_device_id"],
            completion["target_profile_instance_id"],
            completion["profile_build_id"],
        )
        if stage == "sim":
            key = (
                *key,
                completion["latency_method"],
                completion["sim_host_device_id"],
                completion["sim_host_instance_id"],
            )
        records[key] = (completion, protocol_root)
    return list(records.values())


def _run_dirs(protocol_root: Path, relative_dirs: list[str]) -> list[Path]:
    return [protocol_root / relative_dir for relative_dir in relative_dirs]


def _validate_measure_completion(
    completion: dict[str, Any],
    protocol_root: Path,
    spec: dict[str, Any],
) -> tuple[list[Path], list[Path]]:
    if (
        completion["seed_set"] != spec["seed_set"]
        or completion["n_episodes"] != spec["n_episodes"]
        or completion["measure_sessions"] != spec["measure_sessions"]
    ):
        raise ValueError("measure completion does not match the protocol spec")
    fit_dirs = _run_dirs(protocol_root, completion["fit_run_dirs"])
    heldout_dirs = _run_dirs(protocol_root, completion["heldout_run_dirs"])
    fit_manifests = [read_json(run_dir / "manifest.json") for run_dir in fit_dirs]
    heldout_manifests = [read_json(run_dir / "manifest.json") for run_dir in heldout_dirs]
    if (
        len(fit_dirs) != len(spec["measure_sessions"]["fit"])
        or len(heldout_dirs) != len(spec["measure_sessions"]["heldout"])
        or {manifest["measure_session_id"] for manifest in fit_manifests}
        != set(spec["measure_sessions"]["fit"])
        or {manifest["measure_session_id"] for manifest in heldout_manifests}
        != set(spec["measure_sessions"]["heldout"])
        or any(manifest["measure_role"] != "fit" for manifest in fit_manifests)
        or any(manifest["measure_role"] != "heldout" for manifest in heldout_manifests)
        or any(
            manifest["status"] != "completed"
            or manifest["protocol_id"] != completion["protocol_id"]
            or manifest["policy"] != completion["policy"]
            or manifest["target_device_id"] != completion["target_device_id"]
            or manifest["target_profile_instance_id"]
            != completion["target_profile_instance_id"]
            or manifest["seed"] != spec["seed_set"][0]
            or manifest["n_episodes"] != spec["n_episodes"]
            for manifest in [*fit_manifests, *heldout_manifests]
        )
        or any(
            manifest["profile_build_id"] != completion["profile_build_id"]
            for manifest in heldout_manifests
        )
        or any(
            summarize_eval_run(run_dir)["n_episodes"] != spec["n_episodes"]
            for run_dir in [*fit_dirs, *heldout_dirs]
        )
    ):
        raise ValueError("measurement runs do not match their completion marker")
    return fit_dirs, heldout_dirs


def _validate_sim_completion(
    completion: dict[str, Any],
    protocol_root: Path,
    spec: dict[str, Any],
) -> list[Path]:
    if (
        completion["seed_set"] != spec["seed_set"]
        or completion["n_episodes"] != spec["n_episodes"]
        or completion["latency_method"] not in spec["latency_methods"]
        or completion["latency_seed_set"] != spec["latency_seed_set"]
    ):
        raise ValueError("sim completion does not match the protocol spec")
    run_dirs = _run_dirs(protocol_root, completion["run_dirs"])
    manifests = [read_json(run_dir / "manifest.json") for run_dir in run_dirs]
    if (
        len(run_dirs) != len(spec["latency_seed_set"])
        or {manifest["latency_seed"] for manifest in manifests}
        != set(spec["latency_seed_set"])
        or any(
            manifest["status"] != "completed"
            or manifest["protocol_id"] != completion["protocol_id"]
            or manifest["policy"] != completion["policy"]
            or manifest["target_device_id"] != completion["target_device_id"]
            or manifest["target_profile_instance_id"]
            != completion["target_profile_instance_id"]
            or manifest["profile_build_id"] != completion["profile_build_id"]
            or manifest["latency_method"] != completion["latency_method"]
            or manifest["sim_host_device_id"] != completion["sim_host_device_id"]
            or manifest["sim_host_instance_id"] != completion["sim_host_instance_id"]
            or manifest["seed"] != spec["seed_set"][0]
            or manifest["n_episodes"] != spec["n_episodes"]
            for manifest in manifests
        )
        or any(
            summarize_eval_run(run_dir)["n_episodes"] != spec["n_episodes"]
            for run_dir in run_dirs
        )
    ):
        raise ValueError("simulation runs do not match their completion marker")
    return run_dirs


def _row_for_measure_completion(
    spec: dict[str, Any], completion: dict[str, Any]
) -> dict[str, Any]:
    return next(
        row
        for row in spec["rows"]
        if row["policy"] == completion["policy"]
        and row["model_type"] == completion["model_type"]
        and row["env_name"] == completion["env_name"]
    )


def _build_multi_session_experiment1_rows(
    spec: dict[str, Any], root: Path, raw_root: str | Path | None
) -> list[dict[str, Any]]:
    devices = _devices_by_id(spec)
    rows: list[dict[str, Any]] = []
    for completion, protocol_root in _completion_records(spec, root, raw_root, "measure"):
        _validate_measure_completion(completion, protocol_root, spec)
        row_spec = _row_for_measure_completion(spec, completion)
        profile_path = root / completion["profile_path"]
        summary, profile_metadata, notes = _load_profile_summary(profile_path)
        device = devices[completion["target_device_id"]]
        out = {
            "experiment_id": "exp1_latency_profile_summary",
            "panel": "a",
            "policy": row_spec["policy"],
            "env": row_spec["env"],
            "profile_id": device["profile_id"],
            "profile_instance_id": completion["target_profile_instance_id"],
            "profile_build_id": completion["profile_build_id"],
            "profile_path": str(profile_path),
            "hardware": device["display_name"],
            "env_fps": profile_metadata["env_fps"],
            "notes": notes,
            **summary,
        }
        rows.append({column: out[column] for column in EXPERIMENT1_COLUMNS})
    return rows


def _build_multi_session_experiment2_rows(
    spec: dict[str, Any], root: Path, raw_root: str | Path | None
) -> list[dict[str, Any]]:
    devices = _devices_by_id(spec)
    measure_records = {
        (
            completion["target_device_id"],
            completion["target_profile_instance_id"],
            completion["profile_build_id"],
        ): (completion, protocol_root)
        for completion, protocol_root in _completion_records(spec, root, raw_root, "measure")
    }
    rows: list[dict[str, Any]] = []
    for sim_completion, sim_root in _completion_records(spec, root, raw_root, "sim"):
        measure_key = (
            sim_completion["target_device_id"],
            sim_completion["target_profile_instance_id"],
            sim_completion["profile_build_id"],
        )
        measure_completion, measure_root = measure_records[measure_key]
        if (
            sim_completion["profile_artifact_sha256"]
            != measure_completion["profile_artifact_sha256"]
            or sim_completion["profile_fitter_sha256"]
            != measure_completion["profile_fitter_sha256"]
        ):
            raise ValueError("sim completion does not reference the measured profile artifact")
        _, realtime_dirs = _validate_measure_completion(measure_completion, measure_root, spec)
        simulated_dirs = _validate_sim_completion(sim_completion, sim_root, spec)
        realtime = summarize_eval_sessions(realtime_dirs)
        simulated = summarize_eval_sessions(simulated_dirs)
        difference, difference_ci95 = independent_mean_difference_ci95(
            first_mean=realtime["mean_return"],
            first_std=realtime["std_return"],
            first_count=realtime["session_count"],
            second_mean=simulated["mean_return"],
            second_std=simulated["std_return"],
            second_count=simulated["session_count"],
        )
        row_spec = _row_for_measure_completion(spec, measure_completion)
        device = devices[sim_completion["target_device_id"]]
        errors = _error_fields(
            real_value=realtime["mean_return"],
            sim_value=simulated["mean_return"],
            real_length=realtime["mean_length"],
            sim_length=simulated["mean_length"],
        )
        out = {
            "experiment_id": "exp2_same_device_calibration",
            "panel": "b",
            "policy": row_spec["policy"],
            "env": row_spec["env"],
            "profile_id": device["profile_id"],
            "profile_instance_id": sim_completion["target_profile_instance_id"],
            "profile_build_id": sim_completion["profile_build_id"],
            "latency_method": sim_completion["latency_method"],
            "hardware": device["display_name"],
            "sim_host_device_id": sim_completion["sim_host_device_id"],
            "sim_host_instance_id": sim_completion["sim_host_instance_id"],
            "realtime_return": realtime["mean_return"],
            "simulated_return_using_L_d": simulated["mean_return"],
            "realtime_length": realtime["mean_length"],
            "sim_length": simulated["mean_length"],
            "realtime_session_count": realtime["session_count"],
            "simulated_session_count": simulated["session_count"],
            "realtime_return_std": realtime["std_return"],
            "simulated_return_std": simulated["std_return"],
            "realtime_return_ci95": realtime["return_ci95"],
            "simulated_return_ci95": simulated["return_ci95"],
            "sim_minus_real_return": difference,
            "sim_minus_real_ci95": difference_ci95,
            "realtime_length_std": realtime["std_length"],
            "simulated_length_std": simulated["std_length"],
            "realtime_length_ci95": realtime["length_ci95"],
            "simulated_length_ci95": simulated["length_ci95"],
            "realtime_n_episodes": realtime["n_episodes"],
            "simulated_n_episodes": simulated["n_episodes"],
            "n_episodes": spec["n_episodes"],
            "seed_set": _seed_set_label(spec["seed_set"]),
            "notes": "",
            **errors,
        }
        rows.append({column: out[column] for column in EXPERIMENT2_COLUMNS})
    return rows


def build_experiment1_rows(
    spec: dict[str, Any],
    root: str | Path,
    *,
    raw_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    root_path = Path(root)
    if "measure_sessions" in spec:
        return _build_multi_session_experiment1_rows(spec, root_path, raw_root)
    manifests = load_run_manifests(root_path, raw_root=raw_root)
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
                    matching_skipped_notes(
                        manifests,
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
                    "profile_path": str(profile_path),
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
    if "measure_sessions" in spec:
        return _build_multi_session_experiment2_rows(spec, root_path, raw_root)
    manifests = load_run_manifests(root_path, raw_root=raw_root)
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
                    ""
                    if realtime
                    else matching_skipped_notes(
                        manifests,
                        stage="measure",
                        experiment_id=experiment_id,
                        row=row_spec,
                        target_device_id=device["id"],
                        target_profile_instance_id=profile_instance_id,
                    ),
                    _missing_note("sim-same", sim, sim_summary),
                    ""
                    if sim
                    else matching_skipped_notes(
                        manifests,
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
    if "measure_sessions" in spec:
        return []
    manifests = load_run_manifests(root_path, raw_root=raw_root)
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
                        ""
                        if true_realtime
                        else matching_skipped_notes(
                            manifests,
                            stage="measure",
                            experiment_id=experiment_id,
                            row=row_spec,
                            target_device_id=target["id"],
                            target_profile_instance_id=target_profile_instance_id,
                        ),
                        _missing_note("sim-cross", sim_cross, sim_summary),
                        ""
                        if sim_cross
                        else matching_skipped_notes(
                            manifests,
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
    write_csv(outputs["experiment1"], build_experiment1_rows(spec, root, raw_root=raw_root), EXPERIMENT1_COLUMNS)
    write_csv(outputs["experiment2"], build_experiment2_rows(spec, root, raw_root=raw_root), EXPERIMENT2_COLUMNS)
    write_csv(outputs["experiment3"], build_experiment3_rows(spec, root, raw_root=raw_root), EXPERIMENT3_COLUMNS)
    return outputs


def _iter_rows(
    spec: dict[str, Any],
    experiment_id: str,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    experiment = spec.get("experiments", {}).get(experiment_id)
    if experiment is None:
        return []
    return [(experiment_id, experiment, row) for row in spec.get("rows", [])]


def _devices(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return list(spec["devices"])


def _devices_by_id(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(device["id"]): device for device in _devices(spec)}


def _load_profile_summary(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"missing profile: {path}")
    manifest = load_profile_manifest(path)
    e2e_summary = manifest["summary"]
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
