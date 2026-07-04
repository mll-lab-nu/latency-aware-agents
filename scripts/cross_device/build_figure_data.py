"""Build the §4.1 figure/table `data.csv` snapshots from aggregated protocol output.

`paper/.../fig6_cross_device/run.py` and `.../table3_latency_profiles/run.py`
read flat `data.csv` files. This script regenerates those snapshots from the
`aggregate` stage outputs (`experiment{1,2,3}_*.csv`) plus the per-device
`infer_latencies.jsonl` / `e2e_latencies.jsonl` traces, so the figures reproduce end-to-end instead of being
hand-curated.

Usage:
    python -m scripts.cross_device.build_figure_data \
        --tables-dir results/cross_device/tables \
        --profiles-root results/cross_device/profiles \
        --out-fig  paper/.../fig6_cross_device/data.csv \
        --out-table paper/.../table3_latency_profiles/data.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from latency_bench.analysis.cross_device_protocol_tables import (
    load_protocol_manifests,
    load_protocol_spec,
)
from latency_bench.latency.profile import load_latency_samples
from latency_bench.utils.io import load_jsonl_records, write_csv


FIG_COLUMNS = (
    "row_type",
    "panel",
    "policy",
    "env",
    "profile_id",
    "profile_instance_id",
    "hardware",
    "latency_ms",
    "realtime_return",
    "simulated_return_using_L_d",
    "return_abs_error",
    "return_rel_error_pct",
    "realtime_length",
    "sim_length",
    "length_abs_error",
    "length_rel_error_pct",
    "target_device_d",
    "target_profile",
    "target_profile_instance_id",
    "target_device_id",
    "replay_device",
    "replay_host_instance_id",
    "episode_id",
    "episode_return_env",
    "survival_steps",
    "true_realtime_return_on_d",
    "simulated_return_on_replay_using_L_d",
    "true_realtime_length_on_d",
    "simulated_length_on_replay_using_L_d",
    "use_in_main_claim",
    "n_episodes",
    "seed_set",
    "notes",
)

TABLE3_COLUMNS = (
    "policy",
    "env",
    "profile_id",
    "profile_instance_id",
    "hardware",
    "fps",
    "inference_mean_ms",
    "e2e_mean_ms",
    "e2e_mean_frames",
    "n_samples",
    "notes",
)

# Columns shared one-to-one between an experiment CSV and a fig row.
_CALIBRATION_PASSTHROUGH = (
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

_PREDICTION_PASSTHROUGH = (
    "policy",
    "env",
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
    "use_in_main_claim",
    "n_episodes",
    "seed_set",
    "notes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fig_row(**values: Any) -> dict[str, Any]:
    row = {column: "" for column in FIG_COLUMNS}
    row.update(values)
    return row


def build_table3_rows(exp1_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Map experiment-1 summary rows to the Table 3 schema (env_fps -> fps)."""
    rows: list[dict[str, Any]] = []
    for src in exp1_rows:
        if not src.get("e2e_mean_ms"):
            continue  # missing profile; skip rather than emit a blank row
        row = {column: src.get(column, "") for column in TABLE3_COLUMNS}
        row["fps"] = src.get("env_fps", "")
        rows.append(row)
    return rows


def _resolve_latency_trace(profile_manifest: dict[str, Any], *, trace_name: str) -> Path:
    """Locate a latency trace beside a completed measure manifest's profile."""
    trace = Path(str(profile_manifest["profile_path"])).parent / trace_name
    if trace.exists():
        return trace
    raise FileNotFoundError(f"missing latency trace: {trace}")


def _profile_manifest_for_exp1_row(src: dict[str, str], manifests: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        manifest
        for manifest in manifests
        if manifest.get("stage") == "measure"
        and manifest.get("status") == "completed"
        and manifest.get("policy") == src["policy"]
        and manifest.get("env") == src["env"]
        and manifest.get("target_profile_id") == src["profile_id"]
        and manifest["target_profile_instance_id"] == src["profile_instance_id"]
    ]
    if not matches:
        raise FileNotFoundError(
            f"missing measure manifest for {src['policy']} {src['env']} "
            f"{src['profile_id']} {src['profile_instance_id']}"
        )
    return sorted(matches, key=lambda item: str(item.get("completed_at") or item.get("started_at") or ""))[-1]


def build_latency_sample_rows(
    exp1_rows: list[dict[str, str]],
    profiles_root: Path,
    *,
    panel_a_env: str,
    spec: dict[str, Any] | None,
    root: Path | None = None,
    raw_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Emit one panel-(a) row per latency sample, read from per-device traces.

    Traces live at `<model>/<env>/<device>/<instance>/`, so the experiment-1 CSV
    rows (keyed by policy/profile_id) are mapped to model_type/env_name/device_id
    through the protocol spec. Without a spec there is nothing to resolve.
    """
    rows: list[dict[str, Any]] = []
    if spec is None or root is None:
        return rows
    manifests = load_protocol_manifests(root, raw_root=raw_root)
    for src in exp1_rows:
        if src.get("env") != panel_a_env:
            continue
        if not src["e2e_mean_ms"]:
            continue
        profile_manifest = _profile_manifest_for_exp1_row(src, manifests)
        e2e_trace = _resolve_latency_trace(profile_manifest, trace_name="e2e_latencies.jsonl")
        inference_trace = _resolve_latency_trace(profile_manifest, trace_name="infer_latencies.jsonl")
        e2e_records = load_jsonl_records(e2e_trace)
        e2e_samples = [record["latency_ms"] for record in e2e_records]
        for latency_ms in e2e_samples:
            rows.append(
                _fig_row(
                    row_type="e2e_latency_sample",
                    panel="a",
                    policy=src["policy"],
                    env=src["env"],
                    profile_id=src["profile_id"],
                    profile_instance_id=src["profile_instance_id"],
                    hardware=src["hardware"],
                    latency_ms=latency_ms,
                )
            )
        # Overhead = e2e - inference, recorded per action only when the measure
        # run had realtime pipeline profiling enabled. Skip when absent so the
        # frame-quantized legacy traces still build the main figure.
        overhead_values = [record["bench_overhead_ms"] for record in e2e_records if "bench_overhead_ms" in record]
        for overhead_ms in overhead_values:
            rows.append(
                _fig_row(
                    row_type="overhead_latency_sample",
                    panel="a",
                    policy=src["policy"],
                    env=src["env"],
                    profile_id=src["profile_id"],
                    profile_instance_id=src["profile_instance_id"],
                    hardware=src["hardware"],
                    latency_ms=overhead_ms,
                )
            )
        inference_samples = load_latency_samples(inference_trace)
        for latency_ms in inference_samples:
            rows.append(
                _fig_row(
                    row_type="inference_latency_sample",
                    panel="a",
                    policy=src["policy"],
                    env=src["env"],
                    profile_id=src["profile_id"],
                    profile_instance_id=src["profile_instance_id"],
                    hardware=src["hardware"],
                    latency_ms=latency_ms,
                )
            )
    return rows


def build_calibration_rows(exp2_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Map experiment-2 same-device calibration rows to panel-(b) rows."""
    rows: list[dict[str, Any]] = []
    for src in exp2_rows:
        if not src.get("realtime_return") or not src.get("simulated_return_using_L_d"):
            continue
        values = {column: src.get(column, "") for column in _CALIBRATION_PASSTHROUGH}
        rows.append(_fig_row(row_type="calibration", panel="b", **values))
    return rows


def build_prediction_rows(exp3_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Map experiment-3 cross-device rows to panel-(c) rows (target_device -> _d)."""
    rows: list[dict[str, Any]] = []
    for src in exp3_rows:
        if not src.get("true_realtime_return_on_d") or not src.get("simulated_return_on_replay_using_L_d"):
            continue
        values = {column: src.get(column, "") for column in _PREDICTION_PASSTHROUGH}
        rows.append(
            _fig_row(
                row_type="prediction",
                panel="c",
                target_device_d=src.get("target_device", ""),
                **values,
            )
        )
    return rows


def build_realtime_episode_return_rows(
    spec: dict[str, Any],
    *,
    root: Path,
    raw_root: Path | None,
) -> list[dict[str, Any]]:
    """Emit one row per realtime measure episode for return-distribution plots."""
    manifests = load_protocol_manifests(root, raw_root=raw_root)
    rows: list[dict[str, Any]] = []
    experiment_id = "exp2_same_device_calibration"
    experiment = spec["experiments"][experiment_id]
    devices_by_id = {str(device["id"]): device for device in spec["devices"]}
    for row_spec in experiment["rows"]:
        for manifest in manifests:
            if manifest["stage"] != "measure":
                continue
            if manifest["status"] != "completed":
                continue
            if not _manifest_contributes_to(manifest, experiment_id):
                continue
            if manifest["policy"] != row_spec["policy"] or manifest["env"] != row_spec["env"]:
                continue
            device = devices_by_id[str(manifest["target_device_id"])]
            metrics_path = Path(str(manifest["output_dir"])) / "episode_metrics.jsonl"
            for record in load_jsonl_records(metrics_path):
                rows.append(
                    _fig_row(
                        row_type="realtime_episode_return",
                        panel="return_distribution",
                        policy=row_spec["policy"],
                        env=row_spec["env"],
                        profile_id=device["profile_id"],
                        profile_instance_id=manifest["target_profile_instance_id"],
                        hardware=device["display_name"],
                        target_device_id=device["id"],
                        episode_id=record["episode_id"],
                        episode_return_env=record["episode_return_env"],
                        survival_steps=record["survival_steps"],
                        realtime_return=record["episode_return_env"],
                        realtime_length=record["survival_steps"],
                        n_episodes=manifest["n_episodes"],
                        seed_set=",".join(str(item) for item in manifest["seed_set"]),
                        notes=manifest["notes"],
                    )
                )
    return rows


def build_simulated_episode_return_rows(
    spec: dict[str, Any],
    *,
    root: Path,
    raw_root: Path | None,
) -> list[dict[str, Any]]:
    """Emit one row per same-device simulated replay episode."""
    manifests = load_protocol_manifests(root, raw_root=raw_root)
    rows: list[dict[str, Any]] = []
    experiment_id = "exp2_same_device_calibration"
    experiment = spec["experiments"][experiment_id]
    devices_by_id = {str(device["id"]): device for device in spec["devices"]}
    for row_spec in experiment["rows"]:
        for manifest in manifests:
            if manifest["stage"] != "sim_same":
                continue
            if manifest["status"] != "completed":
                continue
            if not _manifest_contributes_to(manifest, experiment_id):
                continue
            if manifest["policy"] != row_spec["policy"] or manifest["env"] != row_spec["env"]:
                continue
            device = devices_by_id[str(manifest["target_device_id"])]
            metrics_path = Path(str(manifest["output_dir"])) / "episode_metrics.jsonl"
            for record in load_jsonl_records(metrics_path):
                rows.append(
                    _fig_row(
                        row_type="simulated_episode_return",
                        panel="return_distribution",
                        policy=row_spec["policy"],
                        env=row_spec["env"],
                        profile_id=device["profile_id"],
                        profile_instance_id=manifest["target_profile_instance_id"],
                        hardware=device["display_name"],
                        target_device_id=device["id"],
                        episode_id=record["episode_id"],
                        episode_return_env=record["episode_return_env"],
                        survival_steps=record["survival_steps"],
                        simulated_return_using_L_d=record["episode_return_env"],
                        sim_length=record["survival_steps"],
                        n_episodes=manifest["n_episodes"],
                        seed_set=",".join(str(item) for item in manifest["seed_set"]),
                        notes=manifest["notes"],
                    )
                )
    return rows


def _manifest_contributes_to(manifest: dict[str, Any], experiment_id: str) -> bool:
    experiment_ids = manifest["experiment_ids"]
    if isinstance(experiment_ids, list):
        return experiment_id in {str(item) for item in experiment_ids}
    return manifest["experiment_id"] == experiment_id


def build_all(
    *,
    tables_dir: Path,
    profiles_root: Path,
    out_fig: Path,
    out_table: Path,
    panel_a_env: str,
    spec: dict[str, Any] | None = None,
    root: Path | None = None,
    raw_root: Path | None = None,
) -> dict[str, int]:
    exp1 = read_csv(tables_dir / "experiment1_latency_profile_summary.csv")
    exp2 = read_csv(tables_dir / "experiment2_same_device_calibration.csv")
    exp3 = read_csv(tables_dir / "experiment3_cross_device_prediction.csv")

    fig_rows = (
        build_latency_sample_rows(
            exp1,
            profiles_root,
            panel_a_env=panel_a_env,
            spec=spec,
            root=root,
            raw_root=raw_root,
        )
        + build_calibration_rows(exp2)
        + build_prediction_rows(exp3)
    )
    if spec is not None and root is not None:
        fig_rows += build_realtime_episode_return_rows(spec, root=root, raw_root=raw_root)
        fig_rows += build_simulated_episode_return_rows(spec, root=root, raw_root=raw_root)
    table_rows = build_table3_rows(exp1)

    write_csv(out_fig, fig_rows, fieldnames=FIG_COLUMNS)
    write_csv(out_table, table_rows, fieldnames=TABLE3_COLUMNS)
    return {"fig_rows": len(fig_rows), "table_rows": len(table_rows)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build §4.1 figure/table data.csv from aggregate protocol output.")
    parser.add_argument("--tables-dir", type=Path, required=True, help="Dir with experiment{1,2,3}_*.csv.")
    parser.add_argument("--profiles-root", type=Path, required=True, help="Root containing explicit latency trace folders.")
    parser.add_argument("--out-fig", type=Path, required=True, help="Output fig6/data.csv path.")
    parser.add_argument("--out-table", type=Path, required=True, help="Output table3/data.csv path.")
    parser.add_argument("--spec", type=Path, help="Protocol spec used to append per-episode realtime returns.")
    parser.add_argument("--root", type=Path, help="Protocol output root used to append per-episode realtime returns.")
    parser.add_argument("--raw-root", type=Path, help="Optional downloaded raw results root.")
    parser.add_argument("--panel-a-env", default="Flappy", help="Env whose latency samples populate panel (a).")
    args = parser.parse_args(argv)

    counts = build_all(
        tables_dir=args.tables_dir,
        profiles_root=args.profiles_root,
        out_fig=args.out_fig,
        out_table=args.out_table,
        panel_a_env=args.panel_a_env,
        spec=None if args.spec is None else load_protocol_spec(args.spec),
        root=args.root,
        raw_root=args.raw_root,
    )
    print(f"wrote {args.out_fig} ({counts['fig_rows']} rows)")
    print(f"wrote {args.out_table} ({counts['table_rows']} rows)")


if __name__ == "__main__":
    main()
