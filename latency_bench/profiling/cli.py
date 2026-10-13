"""Measure deployment latency and fit completed local traces."""
import argparse
from pathlib import Path

from latency_bench.core.config import load_config
from latency_bench.profiling.spec import path_segment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m latency_bench profile", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="measure and fit a deployment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--resume", metavar="RUN_ID", type=path_segment)
    fit = commands.add_parser("fit", help="fit completed local measurement sessions")
    fit.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    # Hardware and numerical dependencies are unnecessary for argument help.
    from latency_bench.profiling.runner import fit_profile, run_profile

    if args.command == "run":
        result = run_profile(load_config(args.config), resume_run_id=args.resume)
    else:
        result = fit_profile(args.run_dir)
    for name, value in result.items():
        print(f"{name}={value}")
