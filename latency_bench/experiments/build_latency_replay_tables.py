from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from latency_bench.analysis.table_builders import (
    build_cross_device_prediction_rows,
    build_latency_profile_summary_rows,
    build_profile_match_rows,
    build_same_device_calibration_rows,
    write_rows_csv,
)


TABLE_OUTPUTS = {
    "profiles": "latency_profile_summary.csv",
    "profile_matches": "profile_match_metrics.csv",
    "same_device_calibration": "same_device_calibration.csv",
    "cross_device_prediction": "cross_device_prediction.csv",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build latency replay analysis CSV tables from existing artifacts.")
    parser.add_argument("--spec", type=Path, required=True, help="Path to analysis table spec JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where CSV tables are written.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV outputs.")
    args = parser.parse_args(argv)

    build_latency_replay_tables(args.spec, args.output_dir, overwrite=args.overwrite)


def build_latency_replay_tables(spec_path: str | Path, output_dir: str | Path, *, overwrite: bool = False) -> None:
    spec = _load_spec(spec_path)
    active_sections = [section for section in TABLE_OUTPUTS if spec.get(section)]
    if not active_sections:
        raise ValueError("No table sections found in spec")

    out_dir = Path(output_dir)
    existing = [out_dir / TABLE_OUTPUTS[section] for section in active_sections if (out_dir / TABLE_OUTPUTS[section]).exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing table outputs without --overwrite: {paths}")

    builders = {
        "profiles": build_latency_profile_summary_rows,
        "profile_matches": build_profile_match_rows,
        "same_device_calibration": build_same_device_calibration_rows,
        "cross_device_prediction": build_cross_device_prediction_rows,
    }
    for section in active_sections:
        rows = builders[section](spec[section])
        write_rows_csv(rows, out_dir / TABLE_OUTPUTS[section])


def _load_spec(spec_path: str | Path) -> dict[str, Any]:
    path = Path(spec_path)
    if not path.exists():
        raise FileNotFoundError(f"Spec file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON spec at {path}: {exc.msg}") from exc
    if not isinstance(spec, dict):
        raise ValueError(f"Spec must be a JSON object: {path}")
    return spec


if __name__ == "__main__":
    main()
