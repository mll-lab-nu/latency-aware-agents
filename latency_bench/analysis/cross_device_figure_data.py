"""Build the §4.1 figure/table `data.csv` snapshots from aggregated protocol output.

`paper/.../fig6_cross_device/run.py` and `.../table3_latency_profiles/run.py`
read flat `data.csv` files. This script regenerates those snapshots from the
`aggregate` stage outputs (`experiment{1,2,3}_*.csv`) plus the per-device
`infer_latencies.jsonl` / `e2e_latencies.jsonl` traces, so the figures reproduce
end-to-end instead of being hand-curated.

Usage:
    python -m latency_bench.analysis.cross_device_figure_data \
        --tables-dir results/cross_device/tables \
        --spec configs/experiments/cross_device_protocol.yaml \
        --root results/cross_device \
        --out-fig  paper/.../fig6_cross_device/data.csv \
        --out-table paper/.../table3_latency_profiles/data.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from latency_bench.analysis.cross_device_protocol_tables import EXPERIMENT2_COLUMNS, EXPERIMENT3_COLUMNS
from latency_bench.analysis.eval_stats import summarize_eval_run
from latency_bench.cross_device.manifest import load_run_manifests, matching_manifests
from latency_bench.cross_device.spec import load_protocol_spec
from latency_bench.latency.profile import load_latency_samples
from latency_bench.utils.io import load_jsonl_records, read_csv, write_csv


FIG_COLUMNS = (
    "row_type",
    "panel",
    "policy",
    "env",
    "profile_id",
    "profile_instance_id",
    "hardware",
    "protocol_id",
    "profile_build_id",
    "latency_method",
    "sim_host_device_id",
    "sim_host_instance_id",
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

SAMPLER_ABLATION_COLUMNS = (
    "policy",
    "env",
    "profile_id",
    "profile_instance_id",
    "profile_build_id",
    "hardware",
    "sim_host_device_id",
    "sim_host_instance_id",
    "latency_method",
    "realtime_session_count",
    "simulated_session_count",
    "realtime_mean_return",
    "simulated_mean_return",
    "sim_minus_real_return",
    "sim_minus_real_ci95",
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
_CALIBRATION_PASSTHROUGH = tuple(
    column for column in EXPERIMENT2_COLUMNS if column not in ("experiment_id", "panel")
)
# target_device is mapped to target_device_d separately.
_PREDICTION_PASSTHROUGH = tuple(
    column for column in EXPERIMENT3_COLUMNS if column not in ("experiment_id", "panel", "target_device")
)


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
    """Locate a latency trace beside the manifest's frozen profile."""
    trace = Path(str(profile_manifest["profile_path"])).parent / trace_name
    if trace.exists():
        return trace
    raise FileNotFoundError(f"missing latency trace: {trace}")


def build_sampler_ablation_rows(
    experiment2_rows: list[dict[str, str]],
    *,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the temporal-versus-IID table from completed protocol summaries."""
    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in experiment2_rows:
        key = (
            row["policy"],
            row["env"],
            row["profile_id"],
            row["profile_instance_id"],
            row["profile_build_id"],
            row["hardware"],
            row["sim_host_device_id"],
            row["sim_host_instance_id"],
        )
        grouped.setdefault(key, {})[row["latency_method"]] = row

    rows: list[dict[str, Any]] = []
    for key, by_method in sorted(grouped.items()):
        if set(by_method) != set(spec["latency_methods"]):
            continue
        for method in spec["latency_methods"]:
            source = by_method[method]
            rows.append(
                {
                    "policy": key[0],
                    "env": key[1],
                    "profile_id": key[2],
                    "profile_instance_id": key[3],
                    "profile_build_id": key[4],
                    "hardware": key[5],
                    "sim_host_device_id": key[6],
                    "sim_host_instance_id": key[7],
                    "latency_method": method,
                    "realtime_session_count": source["realtime_session_count"],
                    "simulated_session_count": source["simulated_session_count"],
                    "realtime_mean_return": source["realtime_return"],
                    "simulated_mean_return": source["simulated_return_using_L_d"],
                    "sim_minus_real_return": source["sim_minus_real_return"],
                    "sim_minus_real_ci95": source["sim_minus_real_ci95"],
                }
            )
    return rows


def build_latency_sample_rows(
    exp1_rows: list[dict[str, str]],
    *,
    panel_a_env: str,
    spec: dict[str, Any] | None,
    root: Path | None = None,
    raw_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Emit one panel-(a) row per latency sample, read from per-device traces.

    Traces live beside each measure run's profile, located through the latest
    completed measure manifest for the exp1 row's device/instance. Without a
    spec there is nothing to resolve.
    """
    rows: list[dict[str, Any]] = []
    if spec is None or root is None:
        return rows
    manifests: list[dict[str, Any]] | None = None
    devices_by_profile_id = {str(device["profile_id"]): device for device in spec["devices"]}
    for src in exp1_rows:
        if src.get("env") != panel_a_env:
            continue
        if not src["e2e_mean_ms"]:
            continue
        if src.get("profile_path"):
            profile_manifest = src
        else:
            if manifests is None:
                manifests = load_run_manifests(root, raw_root=raw_root)
            device = devices_by_profile_id[src["profile_id"]]
            matches = matching_manifests(
                manifests,
                stage="measure",
                experiment_id="exp1_latency_profile_summary",
                row={"policy": src["policy"], "env": src["env"]},
                target_device_id=str(device["id"]),
                target_profile_instance_id=src["profile_instance_id"],
            )
            if not matches:
                raise FileNotFoundError(
                    f"missing measure manifest for {src['policy']} {src['env']} "
                    f"{src['profile_id']} {src['profile_instance_id']}"
                )
            profile_manifest = matches[-1]
        e2e_trace = _resolve_latency_trace(profile_manifest, trace_name="e2e_latencies.jsonl")
        inference_trace = _resolve_latency_trace(profile_manifest, trace_name="infer_latencies.jsonl")
        e2e_records = load_jsonl_records(e2e_trace)
        identity = {
            "policy": src["policy"],
            "env": src["env"],
            "profile_id": src["profile_id"],
            "profile_instance_id": src["profile_instance_id"],
            "profile_build_id": src.get("profile_build_id", ""),
            "hardware": src["hardware"],
        }
        for record in e2e_records:
            rows.append(_fig_row(row_type="e2e_latency_sample", panel="a", latency_ms=record["latency_ms"], **identity))
        # Overhead = e2e - inference, recorded per action by realtime pipeline
        # profiling (always on for measure runs).
        for record in e2e_records:
            rows.append(
                _fig_row(row_type="overhead_latency_sample", panel="a", latency_ms=record["bench_overhead_ms"], **identity)
            )
        for latency_ms in load_latency_samples(inference_trace):
            rows.append(_fig_row(row_type="inference_latency_sample", panel="a", latency_ms=latency_ms, **identity))
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


def build_episode_return_rows(
    spec: dict[str, Any],
    *,
    stage: str,
    root: Path,
    raw_root: Path | None,
) -> list[dict[str, Any]]:
    """Emit one row per episode of completed `measure` or `sim_same` runs.

    Used for the return-distribution plots; `measure` runs fill the realtime
    return/length columns and `sim_same` runs the simulated ones.
    """
    manifests = load_run_manifests(root, raw_root=raw_root)
    rows: list[dict[str, Any]] = []
    row_type = "realtime_episode_return" if stage == "measure" else "simulated_episode_return"
    for row_spec in spec["rows"]:
        for device in spec["devices"]:
            matches = matching_manifests(
                manifests,
                stage=stage,
                experiment_id="exp2_same_device_calibration",
                row=row_spec,
                target_device_id=str(device["id"]),
            )
            for manifest in matches:
                metrics_path = Path(str(manifest["output_dir"])) / "episode_metrics.jsonl"
                for record in load_jsonl_records(metrics_path):
                    values = {
                        "row_type": row_type,
                        "panel": "return_distribution",
                        "policy": row_spec["policy"],
                        "env": row_spec["env"],
                        "profile_id": device["profile_id"],
                        "profile_instance_id": manifest["target_profile_instance_id"],
                        "hardware": device["display_name"],
                        "target_device_id": device["id"],
                        "episode_id": record["episode_id"],
                        "episode_return_env": record["episode_return_env"],
                        "survival_steps": record["survival_steps"],
                        "n_episodes": manifest["n_episodes"],
                        "seed_set": ",".join(str(item) for item in manifest["seed_set"]),
                        "notes": manifest["notes"],
                    }
                    if stage == "measure":
                        values["realtime_return"] = record["episode_return_env"]
                        values["realtime_length"] = record["survival_steps"]
                    else:
                        values["simulated_return_using_L_d"] = record["episode_return_env"]
                        values["sim_length"] = record["survival_steps"]
                    rows.append(_fig_row(**values))
    return rows


def build_all(
    *,
    tables_dir: Path,
    out_fig: Path,
    out_table: Path,
    panel_a_env: str,
    out_sampler_ablation: Path | None = None,
    spec: dict[str, Any] | None = None,
    root: Path | None = None,
    raw_root: Path | None = None,
) -> dict[str, int]:
    exp1 = read_csv(tables_dir / "experiment1_latency_profile_summary.csv")
    exp2 = read_csv(tables_dir / "experiment2_same_device_calibration.csv")
    exp3 = read_csv(tables_dir / "experiment3_cross_device_prediction.csv")

    multi_session_mode = spec is not None and "measure_sessions" in spec
    calibration_rows = (
        [row for row in exp2 if row["latency_method"] == spec["primary_latency_method"]]
        if multi_session_mode
        else exp2
    )
    fig_rows = (
        build_latency_sample_rows(
            exp1,
            panel_a_env=panel_a_env,
            spec=spec,
            root=root,
            raw_root=raw_root,
        )
        + build_calibration_rows(calibration_rows)
        + ([] if multi_session_mode else build_prediction_rows(exp3))
    )
    if not multi_session_mode and spec is not None and root is not None:
        for stage in ("measure", "sim_same"):
            fig_rows += build_episode_return_rows(spec, stage=stage, root=root, raw_root=raw_root)
    table_rows = build_table3_rows(exp1)
    sampler_ablation_rows = (
        []
        if not multi_session_mode
        else build_sampler_ablation_rows(exp2, spec=spec)
    )

    write_csv(out_fig, fig_rows, fieldnames=FIG_COLUMNS)
    write_csv(out_table, table_rows, fieldnames=TABLE3_COLUMNS)
    if out_sampler_ablation is not None:
        write_csv(
            out_sampler_ablation,
            sampler_ablation_rows,
            fieldnames=SAMPLER_ABLATION_COLUMNS,
        )
    return {
        "fig_rows": len(fig_rows),
        "table_rows": len(table_rows),
        "sampler_ablation_rows": len(sampler_ablation_rows),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build §4.1 figure/table data.csv from aggregate protocol output.")
    parser.add_argument("--tables-dir", type=Path, required=True, help="Dir with experiment{1,2,3}_*.csv.")
    parser.add_argument("--out-fig", type=Path, required=True, help="Output fig6/data.csv path.")
    parser.add_argument("--out-table", type=Path, required=True, help="Output table3/data.csv path.")
    parser.add_argument("--out-sampler-ablation", type=Path)
    parser.add_argument("--spec", type=Path, help="Protocol spec used to append per-episode realtime returns.")
    parser.add_argument("--root", type=Path, help="Protocol output root used to append per-episode realtime returns.")
    parser.add_argument("--raw-root", type=Path, help="Optional downloaded raw results root.")
    parser.add_argument("--panel-a-env", default="Flappy", help="Env whose latency samples populate panel (a).")
    args = parser.parse_args(argv)

    counts = build_all(
        tables_dir=args.tables_dir,
        out_fig=args.out_fig,
        out_table=args.out_table,
        out_sampler_ablation=args.out_sampler_ablation,
        panel_a_env=args.panel_a_env,
        spec=None if args.spec is None else load_protocol_spec(args.spec),
        root=args.root,
        raw_root=args.raw_root,
    )
    print(f"wrote {args.out_fig} ({counts['fig_rows']} rows)")
    print(f"wrote {args.out_table} ({counts['table_rows']} rows)")


if __name__ == "__main__":
    main()
