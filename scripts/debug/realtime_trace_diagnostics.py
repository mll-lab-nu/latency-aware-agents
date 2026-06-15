from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.analysis.realtime_trace_diagnostics import summarize_realtime_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize realtime steps/actions/latency trace artifacts.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--first-steps", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(summarize_realtime_trace(args.output_dir, first_steps=args.first_steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
