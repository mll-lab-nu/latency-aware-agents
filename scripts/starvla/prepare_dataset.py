#!/usr/bin/env python3
"""Prepare an existing canonical LeRobot export for StarVLA."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

from latency_bench.integrations.starvla.lerobot_export import convert_canonical_rollout_to_lerobot
from latency_bench.integrations.starvla.task_contracts import prepare_lerobot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("balance", "mikasa"), required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-val", type=Path)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--normalization-source", type=Path)
    return parser


def prepare_balance_bootstrap(raw_dir: Path, output_dir: Path) -> None:
    """Convert Balance teacher rows and freeze the bootstrap task contract."""
    converted = output_dir.with_name(f"_{output_dir.name}_converted")
    convert_canonical_rollout_to_lerobot(
        raw_dir,
        converted,
        force=True,
        action_carrier="native",
        write_starvla_handoff=False,
    )
    converted_val = converted.with_name(f"{converted.name}__val")
    prepare_lerobot(
        "balance",
        converted,
        converted_val if converted_val.exists() else None,
        output_dir,
    )
    shutil.rmtree(converted)
    if converted_val.exists():
        shutil.rmtree(converted_val)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_lerobot(
        args.task,
        args.source_train.expanduser().resolve(),
        args.source_val.expanduser().resolve() if args.source_val is not None else None,
        args.output_train.expanduser().resolve(),
        args.normalization_source.expanduser().resolve()
        if args.normalization_source is not None
        else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
