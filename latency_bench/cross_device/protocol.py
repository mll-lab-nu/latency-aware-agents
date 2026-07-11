"""Command line interface for the cross-device latency protocol.

This module owns argument parsing and high-level subcommand dispatch. The actual
planning, config patching, hardware detection, and execution live in sibling
modules so the CLI stays small.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from latency_bench.analysis.cross_device_protocol_tables import write_protocol_tables
from latency_bench.cross_device.spec import (
    detect_hardware,
    load_protocol_spec,
)
from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.cross_device.configs import _row_base_config
from latency_bench.cross_device.sync import download_profile_build, download_results_tree, hf_env
from latency_bench.cross_device.planning import (
    _build_plan,
    _filtered_spec_for_aggregate,
    _iter_selected_rows,
    _output_root,
    _spec_with_cli_overrides,
)
from latency_bench.cross_device.runner import (
    _print_plan_summary,
    publish_measure,
    run_all_profile_simulations,
    run_multi_session_measure,
    run_stage,
)
from latency_bench.utils.io import read_json


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch one protocol subcommand.

    Args:
        argv: Optional argument vector for tests; uses process argv when None.

    Side effects:
        Loads the protocol spec, writes aggregate tables for `aggregate`, or
        delegates execution-oriented stages to the runner.
    """
    parser = argparse.ArgumentParser(description="Run the cross-device latency experiment protocol.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local hardware, configs, and selected protocol rows.")
    doctor.add_argument("--spec", type=Path, required=True)
    doctor.add_argument("--hardware", default="auto")
    doctor.add_argument("--out", type=Path)
    doctor.add_argument("--profiles-dir", type=Path)
    doctor.add_argument("--target")
    doctor.add_argument("--dry-run", action="store_true")
    _add_selection_flags(doctor)
    _add_override_flags(doctor)

    measure = subparsers.add_parser("measure", help="Run realtime measured eval and build latency profiles.")
    measure.add_argument("--spec", type=Path, required=True)
    measure.add_argument("--hardware", default="auto")
    measure.add_argument("--out", type=Path)
    _add_execution_flags(measure)

    publish = subparsers.add_parser("publish-measure", help="Publish a completed multi-session measure protocol.")
    publish.add_argument("--spec", type=Path, required=True)
    publish.add_argument("--hardware", default="auto")
    publish.add_argument("--out", type=Path)

    sim = subparsers.add_parser(
        "sim",
        help="Run simulated replay: this machine's own profile by default, or --target <device/instance>.",
    )
    sim.add_argument("--spec", type=Path, required=True)
    sim.add_argument("--hardware", default="auto")
    target_group = sim.add_mutually_exclusive_group()
    target_group.add_argument(
        "--target",
        help="Replay another machine's profile: <device_id>/<instance_id>. Omit to use this machine.",
    )
    target_group.add_argument("--all-target-profiles", action="store_true")
    sim.add_argument("--profiles-dir", type=Path)
    sim.add_argument("--latency-method", choices=("temporal", "iid"))
    sim.add_argument("--out", type=Path)
    _add_execution_flags(sim)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate protocol outputs into paper tables.")
    aggregate.add_argument("--spec", type=Path, required=True)
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--raw-root", type=Path)
    aggregate.add_argument("--out", type=Path, required=True)
    aggregate.add_argument("--dry-run", action="store_true")
    aggregate.add_argument("--overwrite", action="store_true")
    aggregate.add_argument("--resume", action="store_true")
    aggregate.add_argument(
        "--no-download",
        action="store_true",
        help="Skip auto-downloading per-machine results from HF before aggregating.",
    )
    aggregate.add_argument("--only-experiment")
    _add_selection_flags(aggregate)
    _add_override_flags(aggregate)

    args = parser.parse_args(argv)
    if args.command == "sim":
        if args.all_target_profiles and args.latency_method is None:
            parser.error("sim --all-target-profiles requires --latency-method")
        if not args.all_target_profiles and args.profiles_dir is None:
            parser.error("sim requires --profiles-dir unless --all-target-profiles is set")
    spec = load_protocol_spec(args.spec)
    if args.command == "doctor":
        run_doctor(args, spec)
    elif args.command == "aggregate":
        spec = _spec_with_cli_overrides(spec, args)
        spec = _filtered_spec_for_aggregate(spec, args)
        raw_root = args.raw_root
        hf = hf_env()
        will_download = raw_root is None and hf is not None and not args.no_download
        if args.dry_run:
            if raw_root is not None:
                print(f"raw_root: {raw_root}")
            elif will_download:
                print(f"would_download_results: {hf['repo_id']} -> {args.root / '_hf_raw'}")
            for filename in (
                "experiment1_latency_profile_summary.csv",
                "experiment2_same_device_calibration.csv",
                "experiment3_cross_device_prediction.csv",
            ):
                print(f"would_write: {args.out / filename}")
            return
        if will_download:
            print(f"downloading results from {hf['repo_id']} ...")
            raw_root = download_results_tree(
                repo_id=hf["repo_id"],
                repo_type=hf["repo_type"],
                local_root=args.root / "_hf_raw",
                revision=hf["revision"],
            )
            print(f"raw_root: {raw_root}")
            if "measure_sessions" in spec:
                completion_paths = Path(raw_root).glob(
                    f"**/cross_device/{spec['protocol_id']}/completion/measure/**/*.json"
                )
                for completion_path in completion_paths:
                    download_profile_build(spec, read_json(completion_path), args.root)
        outputs = write_protocol_tables(
            spec=spec,
            root=args.root,
            raw_root=raw_root,
            out=args.out,
            overwrite=args.overwrite,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
    elif args.command == "publish-measure":
        publish_measure(args, spec)
    elif args.command == "sim":
        if args.all_target_profiles:
            run_all_profile_simulations(args, _spec_with_cli_overrides(spec, args))
        else:
            run_stage(args, spec, "sim_same" if args.target is None else "sim_cross")
    else:
        if "measure_sessions" in spec:
            run_multi_session_measure(args, _spec_with_cli_overrides(spec, args))
        else:
            run_stage(args, spec, "measure")


def _add_selection_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--only-policy")
    parser.add_argument("--only-env")
    parser.add_argument("--only-device")
    parser.add_argument("--max-runs", type=int)


def _add_override_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-kind", choices=["latest", "best"])
    parser.add_argument("--n-episodes", type=int)
    parser.add_argument("--seed-set")
    parser.add_argument("--latency-seed-set")


def _add_execution_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-episodes", type=int, default=1)
    parser.add_argument("--smoke-max-steps", type=int)
    _add_selection_flags(parser)
    _add_override_flags(parser)


def run_doctor(args: argparse.Namespace, spec: dict[str, Any]) -> None:
    """Print local readiness for all protocol stages without running models.

    Args:
        args: Parsed CLI namespace, including hardware filters.
        spec: Loaded cross-device protocol spec.

    Side effects:
        Prints selected rows, hardware mapping, and skip reasons.
    """
    detected = detect_hardware(spec, args.hardware)
    cuda_available = bool(torch.cuda.is_available())
    rows = _iter_selected_rows(spec, args)
    output_root = _output_root(spec, getattr(args, "out", None))
    profiles_dir = Path(args.profiles_dir) if args.profiles_dir is not None else output_root / "profiles"
    plans_by_stage = {
        stage: _build_plan(args, spec, stage, output_root, profiles_dir, detected["device"], for_doctor=True)
        for stage in ("measure", "sim_same", "sim_cross")
    }

    print("Hardware:")
    print(f"  detected: {detected['raw']}")
    print(f"  matched device: {detected['device']['id']}")
    print(f"  display_name: {detected['device'].get('display_name')}")
    print(f"  cuda_available: {cuda_available}")
    fingerprint = collect_hardware_fingerprint()
    gpu = fingerprint.get("gpu", {})
    cpu = fingerprint.get("cpu", {})
    memory = fingerprint.get("memory", {})
    print("  fingerprint:")
    print(f"    gpu: {gpu.get('name')} | vram: {gpu.get('memory.total')} | power_limit: {gpu.get('power.limit')}")
    print(f"    cpu: {cpu.get('model_name')} | logical_cores: {cpu.get('logical_cores')}")
    print(f"    ram: {memory.get('total_gib')} GiB")
    print(f"Output root: {output_root}")
    print("Selected rows:")
    for row in rows:
        row_config = _row_base_config(row)
        print(f"  - policy={row.get('policy')} env={row.get('env')} env_fps={row_config['env']['env_fps']}")
    print("Runnable stages:")
    for stage, plan in plans_by_stage.items():
        _print_plan_summary(stage, plan)
    print("Skipped rows:")
    skipped_any = False
    for stage, plan in plans_by_stage.items():
        for item in plan:
            if item.runnable:
                continue
            skipped_any = True
            print(f"  - stage={stage} policy={item.task.row.get('policy')}: {item.skip_reason}")
    if not skipped_any:
        print("  none")


if __name__ == "__main__":
    main()
