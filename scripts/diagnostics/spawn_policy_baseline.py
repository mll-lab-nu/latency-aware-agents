from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.experiments.calibrate_env_fps_from_saturated_latency import _run_spawned_latency_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure policy latency inside a spawned child process.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=45)
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--period-ms", type=float, default=0.0)
    parser.add_argument("--compare-period-ms", type=float)
    args = parser.parse_args()

    baseline = _run_spawned_latency_probe(
        config_path=args.config,
        samples=args.samples,
        skip=args.skip,
        period_ms=args.period_ms,
    )
    result = {"baseline": _without_samples(baseline)}
    if args.compare_period_ms is not None:
        paced = _run_spawned_latency_probe(
            config_path=args.config,
            samples=args.samples,
            skip=args.skip,
            period_ms=float(args.compare_period_ms),
        )
        result["paced"] = _without_samples(paced)
        result["cadence_sensitive_ratio"] = (
            paced["total_latency_ms"]["median"] / baseline["total_latency_ms"]["median"]
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def _without_samples(payload: dict) -> dict:
    return {
        key: value
        for key, value in payload.items()
        if key != "latency_samples_ms"
    }


if __name__ == "__main__":
    main()
