"""Execution and reporting for planned cross-device tasks.

The runner is the only module that launches evaluation subprocesses. It also
writes generated configs and run manifests for both completed and skipped tasks.
"""
from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.cross_device.config import (
    _eval_command,
    _generated_config,
    _profile_command,
    _write_generated_config,
)
from scripts.cross_device.models import PlannedTask, ProtocolTask, RuntimeSelection
from scripts.cross_device.openvla_deps import _openvla_dependency_errors, _print_openvla_dependency_status
from scripts.cross_device.planning import _build_plan, _output_root, _profiles_dir_for_args
from scripts.cross_device.runtime import _stage_hardware, _validate_runtime_safety, detect_runtime_env


def run_stage(args: argparse.Namespace, spec: dict[str, Any], stage: str) -> None:
    """Plan and execute one non-aggregate protocol stage.

    Args:
        args: Parsed CLI namespace with filters, output root, and execution flags.
        spec: Loaded protocol spec.
        stage: Normalized stage name.

    Side effects:
        Writes generated configs/manifests and launches evaluation subprocesses
        unless `--dry-run` is set.
    """
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
    """Execute one runnable task and record its manifest.

    Args:
        task: Concrete task from the planner.
        spec: Loaded protocol spec.
        detected: Hardware detection result for manifest metadata.
        runtime: Runtime detection result for manifest metadata.
        args: Parsed CLI namespace with resume/dry-run/overwrite flags.
        out: Protocol output root used for the aggregate manifest stream.

    Side effects:
        Writes generated config, per-run manifest, global manifest JSONL, and
        launches the eval/profile subprocesses for real runs.
    """
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


def _base_manifest(
    task: ProtocolTask,
    spec: dict[str, Any],
    detected: dict[str, Any],
    runtime: RuntimeSelection,
    eval_cmd: list[str],
    profile_cmd: list[str] | None,
) -> dict[str, Any]:
    """Build the run manifest payload shared by completed, failed, and skipped tasks.

    Args:
        task: Concrete task.
        spec: Loaded protocol spec.
        detected: Hardware detection result.
        runtime: Runtime selection result.
        eval_cmd: Evaluation command that would run this task.
        profile_cmd: Profile-building command for measure tasks.

    Returns:
        A manifest dictionary ready to write after status fields are finalized.
    """
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


def _print_dry_run_plan(stage: str, plan: list[PlannedTask], spec: dict[str, Any]) -> None:
    """Print commands and generated config paths without launching evaluation.

    Args:
        stage: Normalized stage name.
        plan: Planned tasks for this stage.
        spec: Loaded protocol spec.

    Side effects:
        Writes generated YAML configs so the user can inspect exact runtime
        inputs before starting expensive inference runs.
    """
    print(f"Dry run stage: {stage}")
    for item in plan:
        task = item.task
        print("---")
        print(f"stage: {task.stage}")
        print(f"policy: {task.row.get('policy')}")
        print(f"env: {task.row.get('env')}")
        print(f"experiment_ids: {','.join(task.experiment_ids)}")
        print(f"target_device: {task.target_device.get('id')}")
        print(f"target_profile: {task.target_device.get('profile_id')}")
        print(f"replay_device: {task.replay_device.get('id') if task.replay_device else ''}")
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
        try:
            config = _generated_config(task, spec)
            _write_generated_config(task.generated_config_path, config)
        except Exception as exc:
            print(f"generated_config_error: {exc}")
            continue
        if task.stage == "measure":
            print(f"profile_command: {shlex.join(_profile_command(task, spec, False))}")
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
    return "other"


def _completed_manifest_exists(output_dir: Path) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest.get("status") == "completed"


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
