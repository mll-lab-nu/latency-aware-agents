from __future__ import annotations

import argparse
from pathlib import Path

from latency_bench.cross_device.runner import (
    publish_measure,
    publish_simulation,
    run_measure,
    run_simulation,
)
from latency_bench.cross_device.spec import load_study


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a cross-device sim2real study.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    measure = subparsers.add_parser("measure")
    _add_spec(measure)
    measure.add_argument("--hardware", default="auto")
    measure.add_argument("--resume", action="store_true")

    sim = subparsers.add_parser("sim")
    _add_spec(sim)
    sim.add_argument("--method", required=True)
    sim.add_argument("--resume", action="store_true")

    publish = subparsers.add_parser("publish")
    _add_spec(publish)
    publish.add_argument("--stage", choices=("measure", "sim"), required=True)
    publish.add_argument("--hardware", default="auto")
    publish.add_argument("--method")

    args = parser.parse_args(argv)
    study = load_study(args.spec)
    if args.command == "measure":
        run_measure(study, args.hardware, resume=args.resume)
    elif args.command == "sim":
        run_simulation(study, args.method, resume=args.resume)
    elif args.stage == "measure":
        publish_measure(study, args.hardware)
    else:
        if args.method is None:
            parser.error("publish --stage sim requires --method")
        publish_simulation(study, args.method)


def _add_spec(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", type=Path, required=True)


if __name__ == "__main__":
    main()
