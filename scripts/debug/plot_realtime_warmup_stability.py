from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from latency_bench.executors.realtime_warmup import (
    is_realtime_warmup_stable,
    plot_realtime_warmup,
    realtime_warmup_status,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-output-dir", type=Path)
    parser.add_argument("--latencies-jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.mock_output_dir is not None:
        _plot_mock_cases(args.mock_output_dir)
    if args.latencies_jsonl is not None:
        samples = _read_latency_values(args.latencies_jsonl)
        output = args.output or args.latencies_jsonl.with_name("warmup_stability_from_trace.png")
        plot_realtime_warmup(samples, output, status=realtime_warmup_status(samples))
        print(f"{output}: stable={is_realtime_warmup_stable(samples)} samples={len(samples)}")


def _plot_mock_cases(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "mock_stable_warmup.png": [20.0 + (idx % 5) * 0.04 for idx in range(220)],
        "mock_unstable_warmup.png": [30.0 - idx * 0.04 + (idx % 5) * 0.03 for idx in range(220)],
    }
    for filename, samples in cases.items():
        status = realtime_warmup_status(samples)
        output = output_dir / filename
        plot_realtime_warmup(samples, output, status=status)
        print(f"{output}: stable={status.stable} samples={len(samples)}")


def _read_latency_values(path: Path) -> list[float]:
    return [
        float(json.loads(line)["latency_ms"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


if __name__ == "__main__":
    main()
