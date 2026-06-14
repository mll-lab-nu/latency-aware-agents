"""Command line interface for the cross-device latency protocol.

This module owns argument parsing and high-level subcommand dispatch. The actual
planning, config patching, runtime detection, and execution live in sibling
modules so the CLI stays small.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from latency_bench.analysis.cross_device_protocol_tables import load_protocol_spec, write_protocol_tables
from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from scripts.cross_device.config import _row_base_config
from scripts.cross_device.planning import (
    _build_plan,
    _filtered_spec_for_aggregate,
    _iter_selected_rows,
    _normalize_stage,
    _output_root,
    _spec_with_cli_overrides,
)
from scripts.cross_device.runner import _print_plan_summary, run_stage
from scripts.cross_device.runtime import (
    _runtime_envs,
    detect_hardware,
    detect_runtime_env,
)


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
    doctor.add_argument("--dry-run", action="store_true")
    _add_runtime_flags(doctor)
    _add_selection_flags(doctor)
    _add_override_flags(doctor)

    measure = subparsers.add_parser("measure", help="Run realtime measured eval and build latency profiles.")
    measure.add_argument("--spec", type=Path, required=True)
    measure.add_argument("--hardware", default="auto")
    measure.add_argument("--out", type=Path)
    _add_execution_flags(measure)

    realtime = subparsers.add_parser("realtime", help="Run true realtime evaluation on this hardware.")
    realtime.add_argument("--spec", type=Path, required=True)
    realtime.add_argument("--hardware", default="auto")
    realtime.add_argument("--out", type=Path)
    _add_execution_flags(realtime)

    sim_same = subparsers.add_parser("sim-same", help="Run same-device simulated replay with L_d.")
    sim_same.add_argument("--spec", type=Path, required=True)
    sim_same.add_argument("--hardware", default="auto")
    sim_same.add_argument("--profiles-dir", type=Path, required=True)
    sim_same.add_argument("--out", type=Path)
    _add_execution_flags(sim_same)

    sim_cross = subparsers.add_parser("sim-cross", help="Run cross-device simulated replay on this hardware.")
    sim_cross.add_argument("--spec", type=Path, required=True)
    sim_cross.add_argument("--replay-hardware", default="auto")
    sim_cross.add_argument("--profiles-dir", type=Path, required=True)
    sim_cross.add_argument("--out", type=Path)
    _add_execution_flags(sim_cross)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate protocol outputs into paper tables.")
    aggregate.add_argument("--spec", type=Path, required=True)
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--raw-root", type=Path)
    aggregate.add_argument("--out", type=Path, required=True)
    aggregate.add_argument("--dry-run", action="store_true")
    aggregate.add_argument("--overwrite", action="store_true")
    aggregate.add_argument("--resume", action="store_true")
    _add_runtime_flags(aggregate)
    _add_selection_flags(aggregate)
    _add_override_flags(aggregate)

    args = parser.parse_args(argv)
    spec = load_protocol_spec(args.spec)
    spec = _spec_with_cli_overrides(spec, args)
    if args.command == "doctor":
        run_doctor(args, spec)
    elif args.command == "aggregate":
        spec = _filtered_spec_for_aggregate(spec, args)
        if args.dry_run:
            if args.raw_root is not None:
                print(f"raw_root: {args.raw_root}")
            for filename in (
                "experiment1_latency_profile_summary.csv",
                "experiment2_same_device_calibration.csv",
                "experiment3_cross_device_prediction.csv",
            ):
                print(f"would_write: {args.out / filename}")
            return
        outputs = write_protocol_tables(
            spec=spec,
            root=args.root,
            raw_root=args.raw_root,
            out=args.out,
            overwrite=args.overwrite,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
    else:
        run_stage(args, spec, _normalize_stage(args.command))


def _add_selection_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--only-policy")
    parser.add_argument("--only-env")
    parser.add_argument("--only-device")
    parser.add_argument("--only-experiment")
    parser.add_argument("--max-runs", type=int)


def _add_override_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-kind", choices=["latest", "best"])
    parser.add_argument("--n-episodes", type=int)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--seed-set")


def _add_runtime_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-env", default="auto")
    parser.add_argument("--strict-runtime-env", action="store_true")
    parser.add_argument("--allow-runtime-env-mismatch", action="store_true")


def _add_execution_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-episodes", type=int, default=1)
    parser.add_argument("--smoke-max-steps", type=int)
    _add_runtime_flags(parser)
    _add_selection_flags(parser)
    _add_override_flags(parser)


def run_doctor(args: argparse.Namespace, spec: dict[str, Any]) -> None:
    """Print local readiness for all protocol stages without running models.

    Args:
        args: Parsed CLI namespace, including hardware/runtime filters.
        spec: Loaded cross-device protocol spec.

    Side effects:
        Prints selected rows, runtime/hardware mapping, and skip reasons.
    """
    detected = detect_hardware(spec, args.hardware)
    runtime = detect_runtime_env(spec, args.runtime_env, bool(args.allow_runtime_env_mismatch))
    cuda_available = bool(torch.cuda.is_available())
    rows = list(_iter_selected_rows(spec, args, include_stage=None))
    output_root = _output_root(spec, getattr(args, "out", None))
    profiles_dir = Path(args.profiles_dir) if args.profiles_dir is not None else output_root / "profiles"
    plans_by_stage = {
        stage: _build_plan(args, spec, stage, output_root, profiles_dir, detected["device"], runtime, for_doctor=True)
        for stage in ("measure", "realtime", "sim_same", "sim_cross")
    }

    print("Runtime environment:")
    print(f"  detected: {runtime.detected_raw}")
    print(f"  matched alias: {runtime.matched_alias}")
    print("  available runtime_envs:")
    for alias, env in _runtime_envs(spec).items():
        print(f"    - {alias}: {env.get('conda_env')} ({env.get('description', '')})")
    compatible = sorted({row.get("policy") for _eid, _exp, row in rows if row.get("runtime_env") == runtime.matched_alias})
    print(f"  compatible rows: {', '.join(str(item) for item in compatible) if compatible else 'none'}")
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
    for experiment_id, experiment, row in rows:
        row_config = _row_base_config(row)
        print(
            f"  - experiment={experiment_id} panel={experiment.get('panel')} "
            f"policy={row.get('policy')} env={row.get('env')} env_fps={row_config['env']['env_fps']} "
            f"runtime_env={row.get('runtime_env')}"
        )
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
