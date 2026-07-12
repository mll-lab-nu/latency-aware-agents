from __future__ import annotations

import argparse
from pathlib import Path

from latency_bench.analysis.cross_device_results import write_study_outputs
from latency_bench.cross_device.runner import (
    current_target_ref,
    publish_profile,
    publish_raw,
    publish_simulation,
    run_fit,
    run_measure,
    run_simulation,
)
from latency_bench.cross_device.spec import load_study, load_workload
from latency_bench.cross_device.sync import download_study


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage cross-device artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    target = subparsers.add_parser("target-ref")
    _add_config(target)
    target.add_argument("--hardware", default="auto")

    measure = subparsers.add_parser("measure")
    _add_config(measure)
    measure.add_argument("--hardware", default="auto")
    _add_sessions(measure)
    measure.add_argument("--resume", action="store_true")

    fit = subparsers.add_parser("fit")
    _add_config(fit)
    fit.add_argument("--target-ref", type=Path, required=True)
    fit.add_argument("--profile-revision", required=True)
    _add_sessions(fit)
    fit.add_argument("--resume", action="store_true")

    sim = subparsers.add_parser("sim")
    _add_config(sim)
    _add_simulation_selection(sim)
    sim.add_argument("--resume", action="store_true")

    publish = subparsers.add_parser("publish")
    publish_subparsers = publish.add_subparsers(dest="artifact", required=True)

    publish_raw_parser = publish_subparsers.add_parser("raw")
    _add_config(publish_raw_parser)
    publish_raw_parser.add_argument("--hardware", default="auto")
    _add_sessions(publish_raw_parser)
    publish_raw_parser.add_argument("--resume", action="store_true")

    publish_profile_parser = publish_subparsers.add_parser("profile")
    _add_config(publish_profile_parser)
    publish_profile_parser.add_argument("--profile-ref", type=Path, required=True)
    publish_profile_parser.add_argument("--resume", action="store_true")

    publish_sim_parser = publish_subparsers.add_parser("sim")
    _add_config(publish_sim_parser)
    _add_simulation_selection(publish_sim_parser)
    publish_sim_parser.add_argument("--resume", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--spec", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "aggregate":
        study = load_study(args.spec)
        paths = write_study_outputs(
            study,
            download_study(study),
            Path(study["output_root"]) / "analysis",
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
        return

    workload = load_workload(args.config)
    if args.command == "target-ref":
        print(current_target_ref(workload, args.hardware))
    elif args.command == "measure":
        run_measure(
            workload,
            args.hardware,
            args.sessions,
            resume=args.resume,
        )
    elif args.command == "fit":
        output_ref = run_fit(
            workload,
            args.target_ref,
            args.profile_revision,
            args.sessions,
            resume=args.resume,
        )
        print(output_ref)
    elif args.command == "sim":
        run_simulation(
            workload,
            args.profile_ref,
            args.method,
            args.latency_seeds,
            resume=args.resume,
        )
    elif args.artifact == "raw":
        publish_raw(
            workload,
            args.hardware,
            args.sessions,
            resume=args.resume,
        )
    elif args.artifact == "profile":
        publish_profile(workload, args.profile_ref, resume=args.resume)
    else:
        publish_simulation(
            workload,
            args.profile_ref,
            args.method,
            args.latency_seeds,
            resume=args.resume,
        )


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/cross_device.yaml"),
    )


def _add_sessions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sessions", type=int, nargs="+", required=True)


def _add_simulation_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile-ref", type=Path, required=True)
    parser.add_argument("--method", choices=("temporal", "iid"), required=True)
    parser.add_argument("--latency-seeds", type=int, nargs="+", required=True)


if __name__ == "__main__":
    main()
