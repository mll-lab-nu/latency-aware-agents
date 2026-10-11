"""Dispatch commands without loading unrelated optional integrations."""
import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m latency_bench")
    parser.add_argument("command", choices=("eval", "collect", "convert", "profile"))
    command = parser.parse_args(args[:1]).command
    # Import only the selected capability and its optional dependencies.
    if command == "eval":
        from latency_bench.eval.cli import main as run
    elif command == "profile":
        from latency_bench.profiling.cli import main as run
    elif command == "collect":
        from latency_bench.data.cli import collect_main as run
    else:
        from latency_bench.data.cli import convert_main as run
    run(args[1:])
