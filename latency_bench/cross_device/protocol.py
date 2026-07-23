from __future__ import annotations

import argparse
from pathlib import Path

from latency_bench.analysis.cross_device_results import write_cross_device_outputs
from latency_bench.cross_device.runner import (
    current_target_ref,
    download_raw,
    publish_profile,
    publish_raw,
    publish_simulation,
    run_fit,
    run_measure,
    run_simulation,
)
from latency_bench.cross_device.spec import load_workload
from latency_bench.cross_device.sync import download_analysis_selection


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
    sim.add_argument("--local-profile", action="store_true")
    sim.add_argument("--resume", action="store_true")

    download = subparsers.add_parser("download")
    download_subparsers = download.add_subparsers(dest="artifact", required=True)
    download_raw_parser = download_subparsers.add_parser("raw")
    _add_config(download_raw_parser)
    download_raw_parser.add_argument("--target-ref", type=Path, required=True)
    _add_sessions(download_raw_parser)
    download_raw_parser.add_argument("--overwrite", action="store_true")

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
    _add_publish_mode(publish_profile_parser)

    publish_sim_parser = publish_subparsers.add_parser("sim")
    _add_config(publish_sim_parser)
    _add_simulation_selection(publish_sim_parser)
    _add_publish_mode(publish_sim_parser)

    aggregate = subparsers.add_parser("aggregate")
    _add_config(aggregate)
    aggregate.add_argument(
        "--profile-ref",
        type=Path,
        action="append",
        required=True,
    )
    aggregate.add_argument("--real-sessions", type=int, nargs="+", required=True)
    aggregate.add_argument(
        "--methods",
        choices=("temporal", "iid", "fixed", "normal"),
        nargs="+",
        required=True,
    )
    aggregate.add_argument("--latency-seeds", type=int, nargs="+", required=True)
    aggregate.add_argument("--output-dir", type=Path)

    args = parser.parse_args(argv)
    workload = load_workload(args.config)
    if args.command == "aggregate":
        output_dir = args.output_dir or Path(workload["artifact_root"]) / "analysis"
        root = download_analysis_selection(
            workload,
            args.profile_ref,
            args.real_sessions,
            args.methods,
            args.latency_seeds,
        )
        paths = write_cross_device_outputs(
            workload,
            root,
            output_dir,
            args.profile_ref,
            args.real_sessions,
            args.methods,
            args.latency_seeds,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "target-ref":
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
        simulation_options = {"resume": args.resume}
        if args.local_profile:
            simulation_options["local_profile"] = True
        run_simulation(
            workload,
            args.profile_ref,
            args.method,
            args.latency_seeds,
            **simulation_options,
        )
    elif args.command == "download":
        download_raw(
            workload,
            args.target_ref,
            args.sessions,
            overwrite=args.overwrite,
        )
    elif args.artifact == "raw":
        publish_raw(
            workload,
            args.hardware,
            args.sessions,
            resume=args.resume,
        )
    elif args.artifact == "profile":
        publish_options = {"resume": args.resume}
        if args.overwrite:
            publish_options["overwrite"] = True
        publish_profile(
            workload,
            args.profile_ref,
            **publish_options,
        )
    else:
        publish_options = {"resume": args.resume}
        if args.overwrite:
            publish_options["overwrite"] = True
        publish_simulation(
            workload,
            args.profile_ref,
            args.method,
            args.latency_seeds,
            **publish_options,
        )


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/cross_device.yaml"),
    )


def _add_sessions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sessions", type=int, nargs="+", required=True)


def _add_publish_mode(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")


def _add_simulation_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile-ref", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("temporal", "iid", "fixed", "normal"),
        required=True,
    )
    parser.add_argument("--latency-seeds", type=int, nargs="+", required=True)


if __name__ == "__main__":
    main()
