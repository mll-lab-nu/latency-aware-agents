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

from latency_bench.analysis.cross_device_protocol_tables import slugify
from latency_bench.latency.profile import load_latency_samples
from latency_bench.utils.io import write_csv


FIG_COLUMNS = (
    "row_type",
    "panel",
    "policy",
    "env",
    "profile_id",
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
    "replay_device",
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
    "replay_device",
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


def _resolve_latency_trace(
    profiles_root: Path,
    *,
    policy: str,
    env: str,
    profile_id: str,
    trace_name: str,
) -> Path:
    """Locate a device's latency trace under a flat or nested profiles root.

    Tries the canonical flat layout `<profiles_root>/<policy>/<env>/<id>/` first;
    falls back to globbing so a per-machine HF download root
    (`.../raw/<device>/<runtime>/cross_device/profiles/<policy>/<env>/<id>/`)
    also resolves.
    """
    fixed = profiles_root / slugify(policy) / slugify(env) / profile_id / trace_name
    if fixed.exists():
        return fixed
    pattern = f"**/{slugify(policy)}/{slugify(env)}/{profile_id}/{trace_name}"
    for match in sorted(profiles_root.glob(pattern)):
        if match.is_file():
            return match
    raise FileNotFoundError(f"missing latency trace: {pattern}")


def build_latency_sample_rows(
    exp1_rows: list[dict[str, str]],
    profiles_root: Path,
    *,
    panel_a_env: str,
    max_samples_per_device: int | None,
) -> list[dict[str, Any]]:
    """Emit one panel-(a) row per latency sample, read from per-device traces."""
    rows: list[dict[str, Any]] = []
    for src in exp1_rows:
        if src.get("env") != panel_a_env:
            continue
        if not src["e2e_mean_ms"]:
            continue
        e2e_trace = _resolve_latency_trace(
            profiles_root,
            policy=src["policy"],
            env=src["env"],
            profile_id=str(src["profile_id"]),
            trace_name="e2e_latencies.jsonl",
        )
        inference_trace = _resolve_latency_trace(
            profiles_root,
            policy=src["policy"],
            env=src["env"],
            profile_id=str(src["profile_id"]),
            trace_name="infer_latencies.jsonl",
        )
        e2e_samples = _downsample(load_latency_samples(e2e_trace), max_samples_per_device)
        for latency_ms in e2e_samples:
            rows.append(
                _fig_row(
                    row_type="e2e_latency_sample",
                    panel="a",
                    policy=src["policy"],
                    env=src["env"],
                    profile_id=src["profile_id"],
                    hardware=src["hardware"],
                    latency_ms=latency_ms,
                )
            )
        inference_samples = _downsample(load_latency_samples(inference_trace), max_samples_per_device)
        for latency_ms in inference_samples:
            rows.append(
                _fig_row(
                    row_type="inference_latency_sample",
                    panel="a",
                    policy=src["policy"],
                    env=src["env"],
                    profile_id=src["profile_id"],
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


def _downsample(samples: list[float], cap: int | None) -> list[float]:
    if cap is None or len(samples) <= cap:
        return samples
    step = len(samples) / cap
    return [samples[int(index * step)] for index in range(cap)]


def build_all(
    *,
    tables_dir: Path,
    profiles_root: Path,
    out_fig: Path,
    out_table: Path,
    panel_a_env: str,
    max_samples_per_device: int | None,
) -> dict[str, int]:
    exp1 = read_csv(tables_dir / "experiment1_latency_profile_summary.csv")
    exp2 = read_csv(tables_dir / "experiment2_same_device_calibration.csv")
    exp3 = read_csv(tables_dir / "experiment3_cross_device_prediction.csv")

    fig_rows = (
        build_latency_sample_rows(
            exp1, profiles_root, panel_a_env=panel_a_env, max_samples_per_device=max_samples_per_device
        )
        + build_calibration_rows(exp2)
        + build_prediction_rows(exp3)
    )
    table_rows = build_table3_rows(exp1)

    write_csv(out_fig, fig_rows, fieldnames=FIG_COLUMNS)
    write_csv(out_table, table_rows, fieldnames=TABLE3_COLUMNS)
    return {"fig_rows": len(fig_rows), "table_rows": len(table_rows)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build §4.1 figure/table data.csv from aggregate protocol output.")
    parser.add_argument("--tables-dir", type=Path, required=True, help="Dir with experiment{1,2,3}_*.csv.")
    parser.add_argument("--profiles-root", type=Path, required=True, help="Root of <policy>/<env>/<profile_id>/{infer,e2e}_latencies.jsonl.")
    parser.add_argument("--out-fig", type=Path, required=True, help="Output fig6/data.csv path.")
    parser.add_argument("--out-table", type=Path, required=True, help="Output table3/data.csv path.")
    parser.add_argument("--panel-a-env", default="Flappy", help="Env whose latency samples populate panel (a).")
    parser.add_argument("--max-samples-per-device", type=int, help="Optional cap on panel-(a) samples per device.")
    args = parser.parse_args(argv)

    counts = build_all(
        tables_dir=args.tables_dir,
        profiles_root=args.profiles_root,
        out_fig=args.out_fig,
        out_table=args.out_table,
        panel_a_env=args.panel_a_env,
        max_samples_per_device=args.max_samples_per_device,
    )
    print(f"wrote {args.out_fig} ({counts['fig_rows']} rows)")
    print(f"wrote {args.out_table} ({counts['table_rows']} rows)")


if __name__ == "__main__":
    main()
