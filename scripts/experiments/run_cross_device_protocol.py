#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latency_bench.analysis.cross_device_protocol_tables import (
    load_protocol_spec,
    slugify,
    write_protocol_tables,
)
from latency_bench.core.config import load_config
from latency_bench.policy.openvla_statistics import openvla_dataset_statistics_path


OPENVLA_POLICY_TYPES = {
    "flappy_openvla",
    "demon_attack_openvla",
    "deadly_corridor_openvla",
}
OPENVLA_CHECKPOINT_ENV_BY_POLICY = {
    "flappy_openvla": "FLAPPY_OPENVLA_CHECKPOINT_PATH",
    "demon_attack_openvla": "DEMON_ATTACK_OPENVLA_CHECKPOINT_PATH",
    "deadly_corridor_openvla": "DEADLY_CORRIDOR_OPENVLA_CHECKPOINT_PATH",
}


STAGE_TO_EXPERIMENTS = {
    "measure": ("exp1_latency_profile_summary", "exp2_same_device_calibration", "exp3_cross_device_prediction"),
    "realtime": ("exp2_same_device_calibration", "exp3_cross_device_prediction"),
    "sim_same": ("exp2_same_device_calibration",),
    "sim_cross": ("exp3_cross_device_prediction",),
}


@dataclass(frozen=True)
class RuntimeSelection:
    requested: str
    detected_raw: str | None
    matched_alias: str | None
    matched_conda_env: str | None
    selected_alias: str | None
    allow_all: bool
    allow_mismatch: bool


@dataclass
class ProtocolTask:
    stage: str
    experiment_id: str
    experiment_ids: list[str]
    panel: str
    row: dict[str, Any]
    target_device: dict[str, Any]
    replay_device: dict[str, Any] | None
    seed: int
    seed_set: list[int]
    n_episodes: int
    checkpoint_kind: str
    output_dir: Path
    generated_config_path: Path
    profile_path: Path | None
    profile_output_dir: Path | None
    runtime_env_alias: str | None
    runtime_conda_env: str | None
    smoke: bool = False
    smoke_max_steps: int | None = None


@dataclass
class PlannedTask:
    task: ProtocolTask
    runnable: bool
    skip_reason: str | None = None


def main(argv: list[str] | None = None) -> None:
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

    check_deps = subparsers.add_parser("check-deps", help="Check selected OpenVLA checkpoint dependencies only.")
    check_deps.add_argument("--spec", type=Path, required=True)
    check_deps.add_argument("--hardware", default="auto")
    check_deps.add_argument("--out", type=Path)
    check_deps.add_argument("--profiles-dir", type=Path)
    check_deps.add_argument("--dry-run", action="store_true")
    _add_runtime_flags(check_deps)
    _add_selection_flags(check_deps)
    _add_override_flags(check_deps)

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
    sim_cross.add_argument("--include-controls", action="store_true")
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
    elif args.command == "check-deps":
        run_check_deps(args, spec)
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
    detected = detect_hardware(spec, args.hardware)
    runtime = detect_runtime_env(spec, args.runtime_env, bool(args.allow_runtime_env_mismatch))
    cuda_available = _cuda_available()
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
    print(f"  speed_rank: {detected['device'].get('speed_rank')}")
    print(f"  cuda_available: {cuda_available}")
    print(f"Output root: {output_root}")
    print("Selected rows:")
    for experiment_id, experiment, row in rows:
        print(
            f"  - experiment={experiment_id} panel={experiment.get('panel')} "
            f"policy={row.get('policy')} env={row.get('env')} fps={row.get('fps')} "
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


def run_check_deps(args: argparse.Namespace, spec: dict[str, Any]) -> None:
    rows = _dedup_openvla_rows(_iter_selected_rows(spec, args, include_stage=None))
    if args.max_runs is not None:
        rows = rows[: int(args.max_runs)]

    print("OpenVLA dependency check")
    print("policy | env | base_model | stats_json | unnorm_key | checkpoint | weights | status")
    if not rows:
        print("no selected OpenVLA rows")
        return

    failures = 0
    for _experiment_id, _experiment, row in rows:
        try:
            config = _openvla_dependency_config(row)
            report = _openvla_dependency_report(config)
        except Exception as exc:
            failures += 1
            print(
                f"{row.get('policy')} | {row.get('env')} | error | error | error | error | error | missing: {exc}"
            )
            continue
        if report["status"] != "ok":
            failures += 1
        print(
            f"{row.get('policy')} | {row.get('env')} | {report['base_model']} | {report['stats_json']} | "
            f"{report['unnorm_key']} | {report['checkpoint']} | {report['weights']} | {report['status']}"
        )
    if failures:
        raise SystemExit(1)


def run_stage(args: argparse.Namespace, spec: dict[str, Any], stage: str) -> None:
    out = _output_root(spec, args.out)
    detected = _stage_hardware(spec, args, stage)
    runtime = detect_runtime_env(spec, args.runtime_env, bool(args.allow_runtime_env_mismatch))
    _validate_runtime_safety(args, runtime)
    profiles_dir = _profiles_dir_for_args(args, out)
    plan = _build_plan(args, spec, stage, out, profiles_dir, detected["device"], runtime)
    if args.max_runs is not None:
        plan = plan[: int(args.max_runs)]
    if args.strict_runtime_env:
        mismatches = [
            item
            for item in plan
            if item.skip_reason
            and ("runtime_env mismatch" in item.skip_reason or "runtime_env selection mismatch" in item.skip_reason)
        ]
        if mismatches:
            details = "; ".join(f"{item.task.row.get('policy')}: {item.skip_reason}" for item in mismatches)
            raise SystemExit(f"--strict-runtime-env failed: {details}")
    if args.dry_run:
        _print_dry_run_plan(stage, plan, spec)
        return
    if runtime.allow_mismatch:
        print("WARNING: --allow-runtime-env-mismatch is set; incompatible runtime_env rows may execute.", file=sys.stderr)
    if not plan:
        print("No protocol runs selected.")
        return
    failures: list[dict[str, str]] = []
    for item in plan:
        if not item.runnable:
            _record_skipped_task(item.task, spec, detected, runtime, item.skip_reason or "skipped", out)
            print(f"skipping {item.task.row.get('policy')} {item.task.stage}: {item.skip_reason}")
            continue
        task = item.task
        try:
            print(
                "running "
                f"stage={task.stage} policy={task.row.get('policy')} env={task.row.get('env')} "
                f"output_dir={task.output_dir}"
            )
            _run_task(task, spec, detected, runtime, args, out)
        except Exception as exc:
            failures.append(_failed_run_summary(task, exc))
            print(
                "failed "
                f"stage={task.stage} policy={task.row.get('policy')} env={task.row.get('env')} "
                f"output_dir={task.output_dir}: {exc}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                _print_failed_summary(failures)
                raise
    if failures:
        _print_failed_summary(failures)
        raise SystemExit(1)


def _run_task(
    task: ProtocolTask,
    spec: dict[str, Any],
    detected: dict[str, Any],
    runtime: RuntimeSelection,
    args: argparse.Namespace,
    out: Path,
) -> None:
    if args.resume and _completed_manifest_exists(task.output_dir):
        print(f"resume: skipping completed run {task.output_dir}")
        return

    eval_cmd = _eval_command(task)
    profile_cmd = _profile_command(task, spec, args.overwrite) if task.stage == "measure" else None
    manifest = _base_manifest(task, spec, detected, runtime, eval_cmd, profile_cmd)
    task.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        config = _generated_config(task, spec)
        _write_generated_config(task.generated_config_path, config)
        print(f"generated_config: {task.generated_config_path}")
        print(shlex.join(eval_cmd))
        if profile_cmd is not None:
            print(shlex.join(profile_cmd))
        return

    try:
        config = _generated_config(task, spec)
        _write_generated_config(task.generated_config_path, config)
        dependency_errors = _openvla_dependency_errors(config)
        if dependency_errors:
            raise FileNotFoundError("; ".join(dependency_errors))
        _write_json(task.output_dir / "manifest.json", manifest)
        subprocess.run(eval_cmd, check=True)
        if profile_cmd is not None:
            subprocess.run(profile_cmd, check=True)
        manifest["status"] = "completed"
        manifest["completed_at"] = _now_iso()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = _now_iso()
        manifest["error"] = str(exc)
        manifest["notes"] = _append_note(manifest.get("notes"), str(exc))
        _write_json(task.output_dir / "manifest.json", manifest)
        _append_manifest(out, manifest)
        raise
    _write_json(task.output_dir / "manifest.json", manifest)
    _append_manifest(out, manifest)


def _build_plan(
    args: argparse.Namespace,
    spec: dict[str, Any],
    stage: str,
    out: Path,
    profiles_dir: Path,
    current_device: dict[str, Any],
    runtime: RuntimeSelection,
    *,
    for_doctor: bool = False,
) -> list[PlannedTask]:
    plan_by_key: dict[tuple[Any, ...], PlannedTask] = {}
    seed_set = _seed_set(spec, args)
    checkpoint_kind = str(args.checkpoint_kind or spec.get("checkpoint_kind", "best"))
    for experiment_id, experiment, row in _iter_selected_rows(spec, args, include_stage=stage):
        n_episodes = _n_episodes_for_row_stage(spec, args, row, stage)
        target_devices = _target_devices_for_stage(spec, args, stage, current_device)
        if not target_devices:
            target_devices = [current_device]
        replay_device = current_device if stage == "sim_cross" else None
        for target_device in target_devices:
            for seed in seed_set:
                profile_dir = profiles_dir / slugify(row.get("policy")) / slugify(row.get("env")) / str(target_device["profile_id"])
                profile_path = profile_dir / "profile.json"
                runtime_alias = str(row.get("runtime_env") or "")
                runtime_conda = _runtime_conda_env(spec, runtime_alias)
                task = _make_task(
                    stage=stage,
                    experiment_id=experiment_id,
                    experiment_ids=[experiment_id],
                    panel=str(experiment.get("panel", "")),
                    row=row,
                    target_device=target_device,
                    replay_device=replay_device,
                    seed=int(seed),
                    seed_set=seed_set,
                    n_episodes=n_episodes,
                    checkpoint_kind=checkpoint_kind,
                    out=out,
                    profile_path=profile_path if stage in {"sim_same", "sim_cross"} else None,
                    profile_output_dir=profile_dir if stage == "measure" else None,
                    runtime_env_alias=runtime_alias or None,
                    runtime_conda_env=runtime_conda,
                    smoke=bool(getattr(args, "smoke", False)),
                    smoke_max_steps=getattr(args, "smoke_max_steps", None),
                )
                key = _dedup_key(task, spec)
                skip_reason = _task_skip_reason(
                    task=task,
                    spec=spec,
                    args=args,
                    stage=stage,
                    current_device=current_device,
                    runtime=runtime,
                    for_doctor=for_doctor,
                )
                item = PlannedTask(task=task, runnable=skip_reason is None, skip_reason=skip_reason)
                existing = plan_by_key.get(key)
                if existing is None:
                    plan_by_key[key] = item
                    continue
                if experiment_id not in existing.task.experiment_ids:
                    existing.task.experiment_ids.append(experiment_id)
                if existing.runnable:
                    continue
                if item.runnable:
                    item.task.experiment_ids = existing.task.experiment_ids
                    plan_by_key[key] = item
                elif item.skip_reason and existing.skip_reason and item.skip_reason not in existing.skip_reason:
                    existing.skip_reason = f"{existing.skip_reason}; {item.skip_reason}"
    return list(plan_by_key.values())


def _make_task(
    *,
    stage: str,
    experiment_id: str,
    experiment_ids: list[str],
    panel: str,
    row: dict[str, Any],
    target_device: dict[str, Any],
    replay_device: dict[str, Any] | None,
    seed: int,
    seed_set: list[int],
    n_episodes: int,
    checkpoint_kind: str,
    out: Path,
    profile_path: Path | None,
    profile_output_dir: Path | None,
    runtime_env_alias: str | None,
    runtime_conda_env: str | None,
    smoke: bool = False,
    smoke_max_steps: int | None = None,
) -> ProtocolTask:
    policy_slug = slugify(row.get("policy"))
    env_slug = slugify(row.get("env"))
    stage_dir = stage
    run_id = f"smoke_seed_{seed}_n{n_episodes}" if smoke else f"{checkpoint_kind}_seed_{seed}_n{n_episodes}"
    if stage == "sim_cross" and replay_device is not None:
        device_or_pair = f"{target_device['id']}_replay_{replay_device['id']}"
        output_dir = out / "runs" / stage_dir / policy_slug / env_slug / str(target_device["id"]) / f"replay_{replay_device['id']}" / run_id
    else:
        device_or_pair = str(target_device["id"])
        output_dir = out / "runs" / stage_dir / policy_slug / env_slug / str(target_device["id"]) / run_id
    generated_config_path = out / "generated_configs" / stage_dir / policy_slug / env_slug / device_or_pair / f"{run_id}.yaml"
    return ProtocolTask(
        stage=stage,
        experiment_id=experiment_id,
        experiment_ids=experiment_ids,
        panel=panel,
        row=row,
        target_device=target_device,
        replay_device=replay_device,
        seed=seed,
        seed_set=seed_set,
        n_episodes=n_episodes,
        checkpoint_kind=checkpoint_kind,
        output_dir=output_dir,
        generated_config_path=generated_config_path,
        profile_path=profile_path,
        profile_output_dir=profile_output_dir,
        runtime_env_alias=runtime_env_alias,
        runtime_conda_env=runtime_conda_env,
        smoke=smoke,
        smoke_max_steps=None if smoke_max_steps is None else int(smoke_max_steps),
    )


def _dedup_key(task: ProtocolTask, spec: dict[str, Any]) -> tuple[Any, ...]:
    policy = slugify(task.row.get("policy"))
    env = slugify(task.row.get("env"))
    seed_set = tuple(task.seed_set)
    latency_scope = spec.get("latency_scope", "policy_predict_only")
    profile_id = task.target_device.get("profile_id")
    replay_id = task.replay_device.get("id") if task.replay_device else None
    if task.stage == "measure":
        return (
            "measure",
            policy,
            env,
            task.target_device.get("id"),
            profile_id,
            task.checkpoint_kind,
            seed_set,
            latency_scope,
            task.seed,
        )
    if task.stage == "realtime":
        return ("realtime", policy, env, task.target_device.get("id"), task.checkpoint_kind, seed_set, task.seed)
    if task.stage == "sim_same":
        return (
            "sim_same",
            policy,
            env,
            task.target_device.get("id"),
            profile_id,
            task.checkpoint_kind,
            seed_set,
            task.seed,
        )
    if task.stage == "sim_cross":
        return (
            "sim_cross",
            policy,
            env,
            task.target_device.get("id"),
            replay_id,
            profile_id,
            task.checkpoint_kind,
            seed_set,
            task.seed,
        )
    return (task.stage, policy, env, task.target_device.get("id"), replay_id, task.seed)


def _task_skip_reason(
    *,
    task: ProtocolTask,
    spec: dict[str, Any],
    args: argparse.Namespace,
    stage: str,
    current_device: dict[str, Any],
    runtime: RuntimeSelection,
    for_doctor: bool,
) -> str | None:
    reasons: list[str] = []
    only_devices = _csv_filter(getattr(args, "only_device", None))
    if stage in {"measure", "realtime", "sim_same"} and not _matches_filter(
        _device_filter_candidates(current_device), only_devices
    ):
        reasons.append(f"hardware mismatch: current hardware {current_device.get('id')} not in --only-device")
    if stage == "sim_cross" and not _matches_filter(_device_filter_candidates(task.target_device), only_devices):
        reasons.append(f"hardware mismatch: target hardware {task.target_device.get('id')} not in --only-device")
    if stage == "sim_cross" and task.replay_device is not None and not getattr(args, "include_controls", False):
        if int(task.replay_device.get("speed_rank", -1)) <= int(task.target_device.get("speed_rank", -1)):
            reasons.append("replay device is not faster than target")
    if stage == "sim_cross" and task.replay_device is not None:
        allowed_replay_devices = _experiment_replay_devices(spec, task.experiment_id)
        if allowed_replay_devices and str(task.replay_device.get("id")) not in allowed_replay_devices:
            reasons.append(
                "replay hardware not configured for experiment: "
                f"{task.replay_device.get('id')} not in {','.join(sorted(allowed_replay_devices))}"
            )

    runtime_reason = _runtime_skip_reason(task, runtime)
    if runtime_reason:
        reasons.append(runtime_reason)

    config_reason = _config_dependency_reason(task.row)
    if config_reason:
        reasons.append(config_reason)

    if stage in {"sim_same", "sim_cross"} and task.profile_path is not None and not task.profile_path.exists():
        reasons.append(f"missing profile: {task.profile_path}")

    if reasons:
        return "; ".join(reasons)
    return None


def _runtime_skip_reason(task: ProtocolTask, runtime: RuntimeSelection) -> str | None:
    required = task.runtime_env_alias
    if not required:
        return "missing runtime_env on protocol row"
    if runtime.allow_all:
        if runtime.allow_mismatch:
            return None
        if runtime.matched_alias != required:
            return (
                "runtime_env mismatch: "
                f"requires {required} ({task.runtime_conda_env}), "
                f"current={runtime.detected_raw or 'unknown'}"
            )
        return None
    if runtime.selected_alias != required:
        return f"runtime_env selection mismatch: selected {runtime.selected_alias}, row requires {required}"
    if runtime.allow_mismatch:
        return None
    if runtime.matched_alias != required:
        return (
            "runtime_env mismatch: "
            f"requires {required} ({task.runtime_conda_env}), "
            f"current={runtime.detected_raw or 'unknown'}"
        )
    return None


def _config_dependency_reason(row: dict[str, Any]) -> str | None:
    try:
        path = _base_config_path(row)
    except FileNotFoundError as exc:
        return str(exc)
    if not path.exists():
        return f"missing base_config: {path}"
    try:
        config = load_config(path)
    except Exception as exc:
        return f"config load failed: {exc}"
    return None


def _path_exists_safe(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _openvla_dependency_errors(config: dict[str, Any]) -> list[str]:
    policy = config.get("policy", {})
    policy_type = _policy_type(config)
    if policy_type not in OPENVLA_POLICY_TYPES:
        return []
    errors: list[str] = []
    base_model_value = str(policy.get("base_model_path") or "").strip()
    checkpoint_value = str(policy.get("checkpoint_path") or "").strip()
    unnorm_key = str(policy.get("unnorm_key") or "").strip()
    if not base_model_value:
        errors.append("missing OpenVLA policy.base_model_path")
        base_model_path = None
    else:
        base_model_path = Path(base_model_value)
        if not _path_exists_safe(base_model_path):
            errors.append(f"missing OpenVLA policy.base_model_path: {base_model_path}")
    statistics_path = (
        openvla_dataset_statistics_path(base_model_path, policy_type)
        if base_model_path is not None
        else None
    )
    if statistics_path is not None:
        if not _path_is_file_safe(statistics_path):
            errors.append(f"missing OpenVLA dataset statistics file: {statistics_path}")
        else:
            try:
                statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"failed to read OpenVLA dataset statistics file at {statistics_path}: {exc}")
            else:
                if not isinstance(statistics, dict):
                    errors.append(f"OpenVLA dataset statistics file must contain a JSON object: {statistics_path}")
                elif unnorm_key not in statistics:
                    errors.append(
                        f"missing OpenVLA unnorm_key {unnorm_key!r} in dataset statistics file: {statistics_path}"
                    )
                else:
                    errors.extend(_openvla_action_stats_errors(config, statistics, unnorm_key, statistics_path))
    if not unnorm_key:
        errors.append("missing OpenVLA policy.unnorm_key")
    if not checkpoint_value:
        errors.append("missing OpenVLA policy.checkpoint_path")
    else:
        checkpoint_path = Path(checkpoint_value)
        if not _path_exists_safe(checkpoint_path):
            errors.append(f"missing OpenVLA policy.checkpoint_path: {checkpoint_path}")
        weights_path = checkpoint_path / "actor" / "model_state_dict" / "full_weights.pt"
        if not _path_is_file_safe(weights_path):
            errors.append(f"missing OpenVLA full_weights.pt: {weights_path}")
    return errors


def _openvla_action_stats_errors(
    config: dict[str, Any],
    statistics: dict[str, Any],
    unnorm_key: str,
    statistics_path: Path,
) -> list[str]:
    policy = config.get("policy", {})
    raw_action_dim = policy.get("action_dim")
    if isinstance(raw_action_dim, bool) or not isinstance(raw_action_dim, int) or raw_action_dim <= 0:
        return [f"missing or invalid OpenVLA policy.action_dim for action stats validation: {raw_action_dim!r}"]
    dataset_stats = statistics.get(unnorm_key)
    if not isinstance(dataset_stats, dict):
        return [
            f"OpenVLA stats for {unnorm_key!r} must be an object in dataset statistics file: {statistics_path}"
        ]
    action_stats = dataset_stats.get("action")
    if not isinstance(action_stats, dict):
        return [
            f"OpenVLA stats for {unnorm_key!r} must contain action stats object in dataset statistics file: {statistics_path}"
        ]
    errors: list[str] = []
    for key in ("q01", "q99", "min", "max", "mask"):
        value = action_stats.get(key)
        if value is None:
            errors.append(f"{unnorm_key} action stats missing {key} in dataset statistics file: {statistics_path}")
            continue
        try:
            length = len(value)
        except TypeError:
            errors.append(
                f"{unnorm_key} action stats {key} must be a sequence of length {raw_action_dim}, "
                f"got {type(value).__name__}"
            )
            continue
        if length != raw_action_dim:
            errors.append(
                f"{unnorm_key} action stats {key} length {length} != policy.action_dim {raw_action_dim}"
            )
    return errors


def _openvla_dependency_config(row: dict[str, Any]) -> dict[str, Any]:
    base_config = _base_config_path(row)
    if not base_config.exists():
        raise FileNotFoundError(f"base_config does not exist: {base_config}")
    config = deepcopy(load_config(base_config))
    config.setdefault("policy", {})["type"] = row.get("policy_type", config.get("policy", {}).get("type"))
    config.setdefault("env", {})["name"] = row.get("env_name", config.get("env", {}).get("name"))
    if row.get("fps") is not None:
        fps = _config_fps_value(row["fps"])
        env_cfg = config.setdefault("env", {})
        env_cfg["env_fps"] = fps
        env_cfg["obs_fps"] = fps
    _apply_openvla_env_overrides(config)
    return config


def _openvla_dependency_report(config: dict[str, Any]) -> dict[str, str]:
    policy = config.get("policy", {})
    policy_type = _policy_type(config)
    base_model_value = str(policy.get("base_model_path") or "").strip()
    checkpoint_value = str(policy.get("checkpoint_path") or "").strip()
    unnorm_key = str(policy.get("unnorm_key") or "").strip()
    base_model_path = Path(base_model_value) if base_model_value else None
    stats_path = (
        openvla_dataset_statistics_path(base_model_path, policy_type)
        if base_model_path is not None
        else None
    )
    checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
    weights_path = checkpoint_path / "actor" / "model_state_dict" / "full_weights.pt" if checkpoint_path is not None else None

    base_status = _dependency_path_status(base_model_path, want_file=False)
    stats_status = _dependency_path_status(stats_path, want_file=True)
    checkpoint_status = _dependency_path_status(checkpoint_path, want_file=False)
    weights_status = _dependency_path_status(weights_path, want_file=True)
    unnorm_status = _dependency_unnorm_status(stats_path, unnorm_key)
    errors = _openvla_dependency_errors(config)
    return {
        "base_model": base_status,
        "stats_json": stats_status,
        "unnorm_key": unnorm_status,
        "checkpoint": checkpoint_status,
        "weights": weights_status,
        "status": "ok" if not errors else f"missing: {'; '.join(errors)}",
    }


def _dependency_path_status(path: Path | None, *, want_file: bool) -> str:
    if path is None:
        return "missing"
    exists = _path_is_file_safe(path) if want_file else _path_exists_safe(path)
    return f"ok {path}" if exists else f"missing {path}"


def _dependency_unnorm_status(stats_path: Path | None, unnorm_key: str) -> str:
    if not unnorm_key:
        return "missing"
    if stats_path is None or not _path_is_file_safe(stats_path):
        return f"unknown {unnorm_key}"
    try:
        statistics = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return f"unreadable {unnorm_key}"
    if isinstance(statistics, dict) and unnorm_key in statistics:
        return f"ok {unnorm_key}"
    return f"missing {unnorm_key}"


def _path_is_file_safe(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _generated_config(task: ProtocolTask, spec: dict[str, Any]) -> dict[str, Any]:
    base_config = _base_config_path(task.row)
    if not base_config.exists():
        raise FileNotFoundError(f"Base config does not exist for {task.row.get('policy')} / {task.row.get('env')}: {base_config}")
    config = load_config(base_config)
    config = deepcopy(config)
    config.setdefault("experiment", {})["seed"] = int(task.seed)
    config["experiment"]["num_episodes"] = int(task.n_episodes)
    config.setdefault("evaluation", {})["eval_episodes"] = int(task.n_episodes)
    if task.smoke_max_steps is not None:
        config["evaluation"]["eval_max_steps"] = int(task.smoke_max_steps)
    config.setdefault("logging", {})["output_dir"] = str(task.output_dir)
    config.setdefault("policy", {})["type"] = task.row.get("policy_type", config.get("policy", {}).get("type"))
    config.setdefault("env", {})["name"] = task.row.get("env_name", config.get("env", {}).get("name"))
    _apply_openvla_env_overrides(config)
    if task.row.get("fps") is not None:
        fps = _config_fps_value(task.row["fps"])
        config["env"]["env_fps"] = fps
        config["env"]["obs_fps"] = fps

    if task.stage in {"measure", "realtime"}:
        _patch_realtime_measured_config(config, task, spec)
    elif task.stage in {"sim_same", "sim_cross"}:
        _patch_simulated_replay_config(config, task, spec)
    else:
        raise ValueError(f"Unsupported stage: {task.stage}")
    return config


def _config_fps_value(raw: Any) -> int | float:
    fps = float(raw)
    return int(fps) if fps.is_integer() else fps


def _apply_openvla_env_overrides(config: dict[str, Any]) -> None:
    policy = config.setdefault("policy", {})
    policy_type = _policy_type(config)
    if policy_type not in OPENVLA_POLICY_TYPES:
        return
    base_model_path = os.environ.get("OPENVLA_BASE_MODEL_PATH")
    if base_model_path:
        policy["base_model_path"] = base_model_path
    checkpoint_env = OPENVLA_CHECKPOINT_ENV_BY_POLICY.get(policy_type)
    if checkpoint_env:
        checkpoint_path = os.environ.get(checkpoint_env)
        if checkpoint_path:
            policy["checkpoint_path"] = checkpoint_path


def _policy_type(config: dict[str, Any]) -> str:
    return str(config.get("policy", {}).get("type", "")).strip().lower()


def _patch_realtime_measured_config(config: dict[str, Any], task: ProtocolTask, spec: dict[str, Any]) -> None:
    config.setdefault("executor", {})["mode"] = "realtime"
    latency = config.setdefault("latency", {})
    latency["type"] = "measured"
    latency["device"] = latency.get("device") or "cuda:0"
    latency["sync_cuda"] = latency.get("sync_cuda", True)
    latency["profile_id"] = task.target_device["profile_id"]
    latency["profile_type"] = "rollout_profile"
    latency["latency_scope"] = spec.get("latency_scope", "policy_predict_only")
    latency["target_device"] = task.target_device["id"]
    latency.pop("fixed_latency_ms", None)
    latency.pop("profile_path", None)
    evaluation = config.setdefault("evaluation", {})
    evaluation["eval_parallel_envs"] = 1
    evaluation["eval_latency_values"] = None


def _patch_simulated_replay_config(config: dict[str, Any], task: ProtocolTask, spec: dict[str, Any]) -> None:
    if task.profile_path is None:
        raise ValueError("Simulated replay task requires a profile_path")
    config.setdefault("executor", {})["mode"] = "simulated"
    latency = config.setdefault("latency", {})
    latency["type"] = "empirical_trace_replay"
    latency["profile_path"] = str(task.profile_path)
    latency["trace_replay_order"] = spec.get("trace_replay_order", "sequential")
    latency["loop_trace"] = bool(spec.get("loop_trace", True))
    latency["seed"] = int(task.seed)
    latency["profile_id"] = task.target_device["profile_id"]
    latency["profile_type"] = "rollout_profile"
    latency["latency_scope"] = spec.get("latency_scope", "policy_predict_only")
    latency["target_device"] = task.target_device["id"]
    if task.replay_device is not None:
        latency["replay_host"] = task.replay_device["id"]
    for key in ("fixed_latency_ms", "mean_latency_ms", "std_latency_ms", "min_latency_ms", "max_latency_ms", "trace_path"):
        latency.pop(key, None)
    evaluation = config.setdefault("evaluation", {})
    evaluation["eval_parallel_envs"] = 1
    evaluation["eval_latency_values"] = None


def _write_generated_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _eval_command(task: ProtocolTask) -> list[str]:
    return [
        sys.executable,
        "-m",
        "latency_bench.run",
        "--config",
        str(task.generated_config_path),
        "--checkpoint-kind",
        task.checkpoint_kind,
        "--output-dir",
        str(task.output_dir),
    ]


def _profile_command(task: ProtocolTask, spec: dict[str, Any], overwrite: bool) -> list[str]:
    if task.profile_output_dir is None:
        raise ValueError("Measure task requires profile_output_dir")
    latency_scope = str(spec.get("latency_scope", "policy_predict_only"))
    trace_name = "latencies.jsonl" if latency_scope == "policy_predict_only" else "actions.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "latency_bench.experiments.build_latency_profile_from_trace",
        "--latencies",
        str(task.output_dir / trace_name),
        "--output-dir",
        str(task.profile_output_dir),
        "--profile-id",
        str(task.target_device["profile_id"]),
        "--profile-type",
        "rollout_profile",
        "--latency-scope",
        latency_scope,
        "--policy-id",
        str(task.row.get("policy")),
        "--env-id",
        str(task.row.get("env_name", task.row.get("env"))),
        "--target-device",
        str(task.target_device["id"]),
        "--fps",
        str(task.row["fps"]),
        "--batch-size",
        "1",
        "--n-warmup",
        str(spec.get("n_warmup", 0)),
    ]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def _base_manifest(
    task: ProtocolTask,
    spec: dict[str, Any],
    detected: dict[str, Any],
    runtime: RuntimeSelection,
    eval_cmd: list[str],
    profile_cmd: list[str] | None,
) -> dict[str, Any]:
    replay_device = task.replay_device
    return {
        "protocol_id": spec.get("protocol_id"),
        "stage": task.stage,
        "experiment_id": task.experiment_id,
        "experiment_ids": task.experiment_ids,
        "panel": task.panel,
        "policy": task.row.get("policy"),
        "env": task.row.get("env"),
        "policy_type": task.row.get("policy_type"),
        "env_name": task.row.get("env_name"),
        "target_device_id": task.target_device.get("id"),
        "target_device_display_name": task.target_device.get("display_name"),
        "target_profile_id": task.target_device.get("profile_id"),
        "replay_device_id": None if replay_device is None else replay_device.get("id"),
        "replay_device_display_name": None if replay_device is None else replay_device.get("display_name"),
        "hardware_detected_raw": detected.get("raw"),
        "hardware_normalized": detected["device"].get("id"),
        "runtime_env": task.runtime_env_alias,
        "runtime_conda_env": task.runtime_conda_env,
        "runtime_env_detected_raw": runtime.detected_raw,
        "runtime_env_matched_alias": runtime.matched_alias,
        "profile_path": str(task.profile_path or (task.profile_output_dir / "profile.json" if task.profile_output_dir else "")),
        "output_dir": str(task.output_dir),
        "generated_config_path": str(task.generated_config_path),
        "command": shlex.join(eval_cmd),
        "profile_command": "" if profile_cmd is None else shlex.join(profile_cmd),
        "checkpoint_kind": task.checkpoint_kind,
        "n_episodes": task.n_episodes,
        "smoke": task.smoke,
        "smoke_max_steps": task.smoke_max_steps,
        "seed_set": task.seed_set,
        "seed": task.seed,
        "latency_scope": spec.get("latency_scope", "policy_predict_only"),
        "status": "running",
        "skip_reason": "",
        "error": "",
        "started_at": _now_iso(),
        "completed_at": None,
        "git_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "notes": "",
    }


def _append_manifest(out: Path, manifest: dict[str, Any]) -> None:
    path = out / "manifests" / "runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, sort_keys=True) + "\n")


def _record_skipped_task(
    task: ProtocolTask,
    spec: dict[str, Any],
    detected: dict[str, Any],
    runtime: RuntimeSelection,
    skip_reason: str,
    out: Path,
) -> None:
    manifest = _base_manifest(task, spec, detected, runtime, _eval_command(task), None)
    manifest["status"] = "skipped"
    manifest["completed_at"] = _now_iso()
    manifest["skip_reason"] = skip_reason
    manifest["notes"] = skip_reason
    task.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(task.output_dir / "manifest.json", manifest)
    _append_manifest(out, manifest)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def detect_hardware(spec: dict[str, Any], requested: str = "auto") -> dict[str, Any]:
    devices = _devices(spec)
    if requested != "auto":
        device = normalize_hardware_name(requested, devices)
        if device is None:
            _raise_unknown_hardware(requested, devices)
        return {"raw": requested, "device": device}

    candidates: list[str] = []
    torch_name = _torch_device_name()
    if torch_name:
        candidates.append(torch_name)
    smi_name = _nvidia_smi_device_name()
    if smi_name:
        candidates.append(smi_name)
    env_name = os.environ.get("CROSS_DEVICE_HARDWARE")
    if env_name:
        candidates.append(env_name)
    for candidate in candidates:
        device = normalize_hardware_name(candidate, devices)
        if device is not None:
            return {"raw": candidate, "device": device}
    _raise_unknown_hardware("; ".join(candidates) if candidates else "<none>", devices)


def detect_runtime_env(spec: dict[str, Any], requested: str = "auto", allow_mismatch: bool = False) -> RuntimeSelection:
    detected_raw = _detect_runtime_env_raw()
    matched_alias = _resolve_runtime_alias(spec, detected_raw) if detected_raw else None
    matched_conda_env = _runtime_conda_env(spec, matched_alias) if matched_alias else None
    if requested == "auto":
        selected_alias = matched_alias
        allow_all = False
    elif requested == "all":
        selected_alias = None
        allow_all = True
    else:
        selected_alias = _resolve_runtime_alias(spec, requested)
        if selected_alias is None:
            _raise_unknown_runtime_env(spec, requested)
        allow_all = False
    return RuntimeSelection(
        requested=requested,
        detected_raw=detected_raw,
        matched_alias=matched_alias,
        matched_conda_env=matched_conda_env,
        selected_alias=selected_alias,
        allow_all=allow_all,
        allow_mismatch=allow_mismatch,
    )


def _detect_runtime_env_raw() -> str | None:
    override = os.environ.get("CROSS_DEVICE_RUNTIME_ENV")
    if override:
        return override
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        return conda_env
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        return Path(virtual_env).name
    executable = Path(sys.executable)
    parts = executable.parts
    for marker in ("envs", "env"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    for part in reversed(parts):
        if part in {"latency", "latency-openvla"}:
            return part
    return None


def _runtime_envs(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    envs = spec.get("runtime_envs", {})
    return envs if isinstance(envs, dict) else {}


def _resolve_runtime_alias(spec: dict[str, Any], value: str | None) -> str | None:
    if not value:
        return None
    normalized = _normalize_name(value)
    for alias, env in _runtime_envs(spec).items():
        names = [alias, env.get("conda_env")]
        if any(_normalize_name(name) == normalized for name in names if name):
            return str(alias)
    return None


def _runtime_conda_env(spec: dict[str, Any], alias: str | None) -> str | None:
    if alias is None:
        return None
    env = _runtime_envs(spec).get(alias)
    if not isinstance(env, dict):
        return None
    value = env.get("conda_env")
    return None if value is None else str(value)


def _raise_unknown_runtime_env(spec: dict[str, Any], value: str) -> None:
    known = []
    for alias, env in _runtime_envs(spec).items():
        known.append(str(alias))
        if env.get("conda_env"):
            known.append(str(env["conda_env"]))
    raise SystemExit(f"Unknown --runtime-env {value!r}. Known runtime envs: {', '.join(sorted(known))}")


def _validate_runtime_safety(args: argparse.Namespace, runtime: RuntimeSelection) -> None:
    if runtime.requested == "all" and not args.dry_run and not runtime.allow_mismatch:
        raise SystemExit("--runtime-env all is only allowed with --dry-run unless --allow-runtime-env-mismatch is set")
    if runtime.requested == "auto" and runtime.matched_alias is None and not args.dry_run and not runtime.allow_mismatch:
        raise SystemExit(
            "Could not detect a compatible runtime env. Set CROSS_DEVICE_RUNTIME_ENV, "
            "activate the right conda env, or use --allow-runtime-env-mismatch."
        )


def normalize_hardware_name(name: str, devices: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = _hardware_name_candidates(name)
    for normalized in candidates:
        for device in devices:
            names = [device.get("id"), device.get("display_name"), *device.get("aliases", [])]
            if any(_normalize_name(item) == normalized for item in names if item is not None):
                return device
    for normalized in candidates:
        if not normalized:
            continue
        for device in devices:
            names = [device.get("id"), device.get("display_name"), *device.get("aliases", [])]
            aliases = [_normalize_name(item) for item in names if item is not None]
            if any(_hardware_names_overlap(normalized, alias) for alias in aliases):
                return device
    return None


def _hardware_name_candidates(name: str) -> list[str]:
    raw = str(name)
    parts = [raw, *raw.replace(";", "\n").splitlines()]
    candidates: list[str] = []
    for part in parts:
        normalized = _normalize_name(part)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _hardware_names_overlap(detected: str, known: str) -> bool:
    if len(detected) < 4 or len(known) < 4:
        return False
    return detected in known or known in detected


def _normalize_name(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _raise_unknown_hardware(detected: str, devices: list[dict[str, Any]]) -> None:
    aliases = []
    for device in devices:
        aliases.extend([str(device.get("id")), str(device.get("display_name"))])
        aliases.extend(str(alias) for alias in device.get("aliases", []))
    raise SystemExit(
        "Could not match hardware. "
        f"detected={detected!r}. Known aliases: {', '.join(sorted(set(aliases)))}"
    )


def _torch_device_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        return None
    return None


def _nvidia_smi_device_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    first = result.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _stage_hardware(spec: dict[str, Any], args: argparse.Namespace, stage: str) -> dict[str, Any]:
    if stage == "sim_cross":
        return detect_hardware(spec, args.replay_hardware)
    return detect_hardware(spec, args.hardware)


def _print_dry_run_plan(stage: str, plan: list[PlannedTask], spec: dict[str, Any]) -> None:
    print(f"Dry run stage: {stage}")
    for item in plan:
        task = item.task
        print("---")
        print(f"stage: {task.stage}")
        print(f"policy: {task.row.get('policy')}")
        print(f"env: {task.row.get('env')}")
        print(f"experiment_ids: {','.join(task.experiment_ids)}")
        print(f"target_hardware: {task.target_device.get('id')}")
        print(f"replay_hardware: {task.replay_device.get('id') if task.replay_device else ''}")
        print(f"runtime_env: {task.runtime_env_alias} ({task.runtime_conda_env})")
        print(f"smoke: {str(task.smoke).lower()}")
        print(f"episodes: {task.n_episodes}")
        print(f"status: {'runnable' if item.runnable else 'skipped'}")
        if item.skip_reason:
            print(f"skip_reason: {item.skip_reason}")
        if task.profile_path is not None:
            print(f"profile_path: {task.profile_path}")
        elif task.profile_output_dir is not None:
            print(f"profile_path: {task.profile_output_dir / 'profile.json'}")
        print(f"output_dir: {task.output_dir}")
        print(f"generated_config_path: {task.generated_config_path}")
        print(f"command: {shlex.join(_eval_command(task))}")
        if task.stage == "measure":
            print(f"profile_command: {shlex.join(_profile_command(task, spec, False))}")
        try:
            config = _generated_config(task, spec)
            _write_generated_config(task.generated_config_path, config)
        except Exception as exc:
            print(f"generated_config_error: {exc}")
            continue
        print(f"eval_max_steps: {config.get('evaluation', {}).get('eval_max_steps', '')}")
        _print_openvla_dependency_status(config)
    _print_plan_summary("Summary", plan)


def _print_plan_summary(label: str, plan: list[PlannedTask]) -> None:
    counts = Counter()
    if plan:
        counts["runnable rows"] = 0
    for item in plan:
        if item.runnable:
            counts["runnable rows"] += 1
        else:
            reason = _skip_reason_bucket(item.skip_reason or "skipped")
            counts[f"skipped {reason}"] += 1
    if not counts:
        print(f"{label}: no selected rows")
        return
    print(f"{label}:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")


def _print_openvla_dependency_status(config: dict[str, Any]) -> None:
    if _policy_type(config) not in OPENVLA_POLICY_TYPES:
        return
    policy = config.get("policy", {})
    base_model_path = str(policy.get("base_model_path") or "")
    checkpoint_path = str(policy.get("checkpoint_path") or "")
    unnorm_key = str(policy.get("unnorm_key") or "")
    stats_path = (
        str(openvla_dataset_statistics_path(Path(base_model_path), _policy_type(config)))
        if base_model_path
        else ""
    )
    print(f"resolved_base_model_path: {base_model_path}")
    print(f"resolved_dataset_statistics_path: {stats_path}")
    print(f"resolved_checkpoint_path: {checkpoint_path}")
    print(f"resolved_unnorm_key: {unnorm_key}")
    dependency_errors = _openvla_dependency_errors(config)
    if dependency_errors:
        print(f"openvla_dependency_status: missing ({'; '.join(dependency_errors)})")
    else:
        print("openvla_dependency_status: ok")


def _skip_reason_bucket(reason: str) -> str:
    if "runtime_env mismatch" in reason or "runtime_env selection mismatch" in reason:
        return "runtime_env mismatch"
    if "replay hardware not configured" in reason:
        return "replay hardware not configured"
    if "missing profile" in reason:
        return "missing profile"
    if "missing base_config" in reason or "config load failed" in reason:
        return "missing config"
    if "checkpoint" in reason or "base_model_path" in reason:
        return "missing checkpoint"
    if "hardware mismatch" in reason:
        return "hardware mismatch"
    if "not faster" in reason:
        return "replay not faster"
    return "other"


def _iter_selected_rows(
    spec: dict[str, Any],
    args: argparse.Namespace,
    *,
    include_stage: str | None,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    allowed_experiments = None if include_stage is None else set(STAGE_TO_EXPERIMENTS[include_stage])
    only_experiments = _csv_filter(getattr(args, "only_experiment", None))
    only_policies = _csv_filter(getattr(args, "only_policy", None))
    only_envs = _csv_filter(getattr(args, "only_env", None))
    runtime_filter = _runtime_row_filter(spec, getattr(args, "runtime_env", "auto"))
    output: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for experiment_id, experiment in spec.get("experiments", {}).items():
        if allowed_experiments is not None and experiment_id not in allowed_experiments:
            continue
        if not _matches_filter(_experiment_filter_candidates(experiment_id), only_experiments):
            continue
        for row in experiment.get("rows", []):
            if runtime_filter and not _matches_filter(_runtime_filter_candidates(spec, row), runtime_filter):
                continue
            if not _matches_filter(_policy_filter_candidates(row), only_policies):
                continue
            if not _matches_filter(_env_filter_candidates(row), only_envs):
                continue
            output.append((experiment_id, experiment, row))
    return output


def _dedup_openvla_rows(
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    output: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in rows:
        _experiment_id, _experiment, row = item
        policy_type = str(row.get("policy_type") or "").strip().lower()
        if policy_type not in OPENVLA_POLICY_TYPES:
            continue
        key = (slugify(row.get("policy")), slugify(row.get("env")), policy_type)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _target_devices_for_stage(
    spec: dict[str, Any],
    args: argparse.Namespace,
    stage: str,
    current_device: dict[str, Any],
) -> list[dict[str, Any]]:
    only_devices = _csv_filter(getattr(args, "only_device", None))
    if stage in {"measure", "realtime", "sim_same"}:
        if not _matches_filter(_device_filter_candidates(current_device), only_devices):
            return []
        return [current_device]
    devices = _devices(spec)
    if only_devices:
        devices = [device for device in devices if _matches_filter(_device_filter_candidates(device), only_devices)]
    return devices


def _experiment_replay_devices(spec: dict[str, Any], experiment_id: str) -> set[str]:
    experiment = spec.get("experiments", {}).get(experiment_id, {})
    if not isinstance(experiment, dict):
        return set()
    return {str(item) for item in experiment.get("replay_devices", [])}


def _devices(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [device for device in spec.get("devices", []) if isinstance(device, dict)]


def _seed_set(spec: dict[str, Any], args: argparse.Namespace) -> list[int]:
    raw = args.seed_set if getattr(args, "seed_set", None) is not None else spec.get("seed_set", [0])
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    return [int(item) for item in items]


def _profiles_dir_for_args(args: argparse.Namespace, out: Path) -> Path:
    raw = getattr(args, "profiles_dir", None)
    if raw is not None:
        return Path(raw)
    if bool(getattr(args, "smoke", False)):
        return out / "profiles_smoke"
    return out / "profiles"


def _n_episodes_for_row_stage(
    spec: dict[str, Any],
    args: argparse.Namespace,
    row: dict[str, Any],
    stage: str,
) -> int:
    if bool(getattr(args, "smoke", False)):
        return int(getattr(args, "smoke_episodes", 1))
    if getattr(args, "n_episodes", None) is not None:
        return int(args.n_episodes)
    stage_overrides = row.get("stage_overrides", {})
    if isinstance(stage_overrides, dict):
        stage_cfg = stage_overrides.get(stage)
        if stage_cfg is None:
            stage_cfg = stage_overrides.get(stage.replace("_", "-"))
        if isinstance(stage_cfg, dict) and stage_cfg.get("n_episodes") is not None:
            return int(stage_cfg["n_episodes"])
    if row.get("n_episodes") is not None:
        return int(row["n_episodes"])
    return int(spec.get("n_episodes", 1))


def _spec_with_cli_overrides(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = deepcopy(spec)
    if getattr(args, "seed_set", None) is not None:
        updated["seed_set"] = _seed_set(spec, args)
    if getattr(args, "n_episodes", None) is not None:
        updated["n_episodes"] = int(args.n_episodes)
    if getattr(args, "checkpoint_kind", None) is not None:
        updated["checkpoint_kind"] = str(args.checkpoint_kind)
    return updated


def _filtered_spec_for_aggregate(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = deepcopy(spec)
    only_experiments = _csv_filter(getattr(args, "only_experiment", None))
    only_policies = _csv_filter(getattr(args, "only_policy", None))
    only_envs = _csv_filter(getattr(args, "only_env", None))
    only_devices = _csv_filter(getattr(args, "only_device", None))
    runtime_filter = _runtime_row_filter(updated, getattr(args, "runtime_env", "auto"))
    if only_devices:
        updated["devices"] = [
            device for device in _devices(updated) if _matches_filter(_device_filter_candidates(device), only_devices)
        ]
    experiments = updated.get("experiments", {})
    if isinstance(experiments, dict):
        for experiment_id in list(experiments):
            if not _matches_filter(_experiment_filter_candidates(experiment_id), only_experiments):
                experiments.pop(experiment_id)
                continue
            experiment = experiments[experiment_id]
            if not isinstance(experiment, dict):
                continue
            rows = []
            for row in experiment.get("rows", []):
                if runtime_filter and not _matches_filter(_runtime_filter_candidates(updated, row), runtime_filter):
                    continue
                if not _matches_filter(_policy_filter_candidates(row), only_policies):
                    continue
                if not _matches_filter(_env_filter_candidates(row), only_envs):
                    continue
                rows.append(row)
            experiment["rows"] = rows
    return updated


def _csv_filter(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _matches_filter(candidates: list[Any], filters: set[str]) -> bool:
    if not filters:
        return True
    candidate_tokens = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_tokens.update(_filter_token_variants(candidate))
    filter_tokens = set()
    for item in filters:
        filter_tokens.update(_filter_token_variants(item))
    return bool(candidate_tokens & filter_tokens)


def _filter_token_variants(value: Any) -> set[str]:
    raw = str(value).strip()
    if not raw:
        return set()
    return {raw, raw.lower(), slugify(raw), _normalize_name(raw)}


def _policy_filter_candidates(row: dict[str, Any]) -> list[Any]:
    return [row.get("policy"), row.get("policy_type")]


def _env_filter_candidates(row: dict[str, Any]) -> list[Any]:
    return [row.get("env"), row.get("env_name")]


def _experiment_filter_candidates(experiment_id: str) -> list[Any]:
    return [experiment_id]


def _device_filter_candidates(device: dict[str, Any]) -> list[Any]:
    return [
        device.get("id"),
        device.get("display_name"),
        device.get("profile_id"),
        *device.get("aliases", []),
    ]


def _runtime_row_filter(spec: dict[str, Any], requested: Any) -> set[str]:
    value = str(requested or "auto")
    if value in {"auto", "all"}:
        return set()
    alias = _resolve_runtime_alias(spec, value)
    return {alias or value}


def _runtime_filter_candidates(spec: dict[str, Any], row: dict[str, Any]) -> list[Any]:
    alias = row.get("runtime_env")
    return [alias, _runtime_conda_env(spec, str(alias)) if alias else None]


def _base_config_path(row: dict[str, Any]) -> Path:
    value = row.get("base_config")
    if value is None or str(value).strip().upper().startswith("TODO"):
        raise FileNotFoundError(f"Missing base_config for {row.get('policy')} / {row.get('env')}: {value}")
    return Path(str(value))


def _missing_base_configs(rows: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> list[str]:
    missing = []
    for _experiment_id, _experiment, row in rows:
        try:
            path = _base_config_path(row)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        if not path.exists():
            missing.append(f"{row.get('policy')} / {row.get('env')}: {path}")
    return missing


def _checkpoint_notes(rows: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> list[str]:
    notes = []
    for _experiment_id, _experiment, row in rows:
        try:
            path = _base_config_path(row)
        except FileNotFoundError:
            continue
        if not path.exists():
            continue
        try:
            config = load_config(path)
        except Exception as exc:
            notes.append(f"{row.get('policy')} / {row.get('env')}: config load failed: {exc}")
            continue
        policy = config.get("policy", {})
        explicit_checkpoint = str(policy.get("checkpoint_path") or "").strip()
        if explicit_checkpoint:
            for key in ("checkpoint_path", "base_model_path"):
                value = str(policy.get(key) or "").strip()
                if value and not Path(value).exists():
                    notes.append(f"{row.get('policy')} / {row.get('env')}: missing policy.{key}: {value}")
            continue
        backend = config.get("backend", {})
        train_dir = backend.get("train_dir")
        experiment_name = config.get("experiment", {}).get("name")
        if train_dir and experiment_name:
            candidate = Path(str(train_dir)) / str(experiment_name)
            if not candidate.exists():
                notes.append(
                    f"{row.get('policy')} / {row.get('env')}: checkpoint run dir not found: {candidate}"
                )
    return notes


def _output_root(spec: dict[str, Any], out: Path | None) -> Path:
    return Path(out) if out is not None else Path(str(spec.get("output_root", "results/cross_device")))


def _completed_manifest_exists(output_dir: Path) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest.get("status") == "completed"


def _normalize_stage(stage: str) -> str:
    return stage.replace("-", "_")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip()


def _append_note(existing: Any, note: str) -> str:
    if existing:
        return f"{existing}; {note}"
    return note


def _failed_run_summary(task: ProtocolTask, exc: BaseException) -> dict[str, str]:
    return {
        "stage": str(task.stage),
        "policy": str(task.row.get("policy")),
        "env": str(task.row.get("env")),
        "output_dir": str(task.output_dir),
        "error": str(exc),
    }


def _print_failed_summary(failures: list[dict[str, str]]) -> None:
    if not failures:
        return
    print("FAILED RUNS:", file=sys.stderr)
    for failure in failures:
        print(
            "  - "
            f"stage={failure['stage']} "
            f"policy={failure['policy']} "
            f"env={failure['env']} "
            f"output_dir={failure['output_dir']} "
            f"error={failure['error']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
