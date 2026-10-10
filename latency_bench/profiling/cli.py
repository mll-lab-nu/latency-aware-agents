"""Measure from an ordinary run config and operate on explicit profile artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from latency_bench.core.config import load_config
from latency_bench.profiling.profile_resolver import list_remote_profiles, resolve_remote_profile
from latency_bench.profiling.runner import fit_profile, publish_profile, run_profile, run_simulation
from latency_bench.profiling.spec import path_segment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="measure and fit a deployment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--resume", metavar="RUN_ID", type=path_segment)
    run.add_argument("--publish", action="store_true")

    fit = commands.add_parser("fit", help="fit completed sessions without running the model")
    fit.add_argument("--run-dir", type=Path, required=True)

    publish = commands.add_parser("publish", help="publish a completed local run")
    publish.add_argument("--run-dir", type=Path, required=True)
    publish.add_argument("--repo", required=True)

    listing = commands.add_parser("list", help="list published profiles")
    listing.add_argument("--repo", required=True)
    listing.add_argument("--model")
    listing.add_argument("--gpu")
    listing.add_argument("--workload")

    resolve = commands.add_parser("resolve", help="download a complete profile revision")
    resolve.add_argument("--repo", required=True)
    resolve.add_argument("--ref", required=True, help="model/deployment/workload/instance")
    resolve.add_argument("--output-dir", type=Path, required=True)

    simulate = commands.add_parser("simulate", help="replay a local profile with its measured model")
    simulate.add_argument("--config", type=Path, required=True)
    simulate.add_argument("--profile-path", type=Path, required=True)
    simulate.add_argument("--method", choices=("zero", "temporal", "iid", "fixed", "normal"), required=True)
    simulate.add_argument("--seeds", type=int, nargs="+", required=True)
    simulate.add_argument("--publish", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_profile(load_config(args.config), resume_run_id=args.resume, publish=args.publish)
    elif args.command == "fit":
        result = fit_profile(args.run_dir)
    elif args.command == "publish":
        result = publish_profile(args.run_dir, repo_id=args.repo)
    elif args.command == "list":
        for ref in list_remote_profiles(args.repo, model_id=args.model, gpu_class=args.gpu, workload_id=args.workload):
            print(ref.as_posix())
        return
    elif args.command == "resolve":
        path = resolve_remote_profile(args.repo, args.ref, output_dir=args.output_dir)
        result = {"PROFILE_PATH": str(path)}
    else:
        result = run_simulation(load_config(args.config), profile_path=args.profile_path,
                                method=args.method, latency_seeds=args.seeds, publish=args.publish)
    for name, value in result.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
