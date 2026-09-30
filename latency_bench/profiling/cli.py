"""Profile CLI; invoke from the repository root or through scripts/profile.sh."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config
from latency_bench.profiling.profile_resolver import (
    list_remote_profiles,
    resolve_model_artifacts,
    resolve_remote_profile,
)
from latency_bench.profiling.runner import (
    current_deployment,
    publish_current_profile,
    run_profile,
    run_simulation,
)
from latency_bench.profiling.spec import load_profile_spec


REGISTRY_PATH = Path("configs/profiles.yaml")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure and replay deployment latency profiles.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="measure and fit one profile")
    _add_preset(run)
    source = run.add_mutually_exclusive_group()
    source.add_argument("--model-path", type=Path)
    source.add_argument(
        "--hf-artifacts",
        action="store_true",
        help="resolve the Hugging Face artifact coordinates instead of local defaults",
    )
    run.add_argument(
        "--hf-repo-id",
        help="override the Hugging Face model repo that stores the checkpoint",
    )
    run.add_argument(
        "--hf-path",
        type=_repo_relative_path,
        help="override the subdirectory within the Hugging Face repo",
    )
    run.add_argument(
        "--hf-checkpoint",
        type=_repo_relative_path,
        help="override the checkpoint path within the selected Hugging Face repo",
    )
    run.add_argument(
        "--hf-model-config",
        type=_repo_relative_path,
        help="override the model config path within the selected Hugging Face repo",
    )
    run.add_argument("--resume", metavar="RUN_ID", type=_path_segment)
    run.add_argument("--publish", action="store_true")

    publish = commands.add_parser("publish", help="publish the current local profile")
    _add_preset(publish)

    listing = commands.add_parser("list", help="list published profiles")
    listing.add_argument("--model")
    listing.add_argument("--gpu")
    listing.add_argument("--workload")

    resolve = commands.add_parser("resolve", help="download one exact published profile")
    _add_preset(resolve)
    resolve.add_argument("--gpu", type=_path_segment)
    resolve.add_argument("--instance", type=_path_segment)

    simulate = commands.add_parser("simulate", help="replay one exact published profile")
    _add_preset(simulate)
    simulate.add_argument("--gpu", type=_path_segment, required=True)
    simulate.add_argument("--instance", type=_path_segment, required=True)
    simulate.add_argument(
        "--method",
        choices=("zero", "temporal", "iid", "fixed", "normal"),
        required=True,
    )
    simulate.add_argument("--seeds", type=int, nargs="+", required=True)
    simulate.add_argument("--publish", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "list":
        registry = load_config(REGISTRY_PATH)
        refs = list_remote_profiles(
            registry,
            model_id=args.model,
            gpu_class=args.gpu,
            workload_id=args.workload,
        )
        for ref in refs:
            print(ref.as_posix())
        return

    spec = load_profile_spec(REGISTRY_PATH, args.model, args.workload)
    if args.command == "run":
        if args.model_path is not None and _has_hf_artifact_overrides(args):
            parser.error("--model-path cannot be combined with Hugging Face artifact overrides")
        spec = _apply_hf_artifact_overrides(spec, args)
        spec = resolve_model_artifacts(
            spec,
            model_path=args.model_path,
            hf_artifacts=args.hf_artifacts or _has_hf_artifact_overrides(args),
        )
        result = run_profile(spec, resume_run_id=args.resume, publish=args.publish)
        _print_result(result)
    elif args.command == "publish":
        result = publish_current_profile(spec)
        _print_result(result)
    elif args.command == "resolve":
        if (args.gpu is None) != (args.instance is None):
            parser.error("--gpu and --instance must be provided together")
        if args.gpu is None:
            deployment = current_deployment(spec)
            gpu_class = deployment["gpu_class"]
            instance_id = deployment["instance_id"]
        else:
            gpu_class = args.gpu
            instance_id = args.instance
        ref, profile_path = resolve_remote_profile(
            spec,
            gpu_class=gpu_class,
            instance_id=instance_id,
        )
        print(f"PROFILE_REF={ref.as_posix()}")
        print(f"PROFILE_PATH={profile_path}")
    else:
        result = run_simulation(
            spec,
            gpu_class=args.gpu,
            instance_id=args.instance,
            method=args.method,
            latency_seeds=args.seeds,
            publish=args.publish,
        )
        _print_result(result)


def _add_preset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True)


def _path_segment(value: str) -> str:
    if Path(value).name != value or value in {".", ".."}:
        raise argparse.ArgumentTypeError("must be one path segment")
    return value


def _repo_relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise argparse.ArgumentTypeError("must be a relative path within the repo")
    return path.as_posix()


def _has_hf_artifact_overrides(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, field) is not None
        for field in ("hf_repo_id", "hf_path", "hf_checkpoint", "hf_model_config")
    )


def _apply_hf_artifact_overrides(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not _has_hf_artifact_overrides(args):
        return spec

    artifact = dict(spec["model_artifact"])
    if args.hf_repo_id is not None:
        artifact["hf_repo"] = args.hf_repo_id
    if args.hf_path is not None:
        artifact["hf_path"] = args.hf_path
    if args.hf_checkpoint is not None:
        artifact["checkpoint"] = args.hf_checkpoint
    if args.hf_model_config is not None:
        artifact["model_config"] = args.hf_model_config

    resolved = dict(spec)
    resolved["model_artifact"] = artifact
    return resolved


def _print_result(result: dict[str, str]) -> None:
    for name, value in result.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
