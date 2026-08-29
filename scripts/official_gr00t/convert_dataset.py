#!/usr/bin/env python3
"""Convert canonical rollout parquet files for Isaac-GR00T NEW_EMBODIMENT."""

from __future__ import annotations

import argparse
import json

from latency_bench.data.official_gr00t import (
    convert_canonical_rollout_to_official_gr00t,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_export_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = convert_canonical_rollout_to_official_gr00t(
        args.raw_export_dir,
        args.output_dir,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
