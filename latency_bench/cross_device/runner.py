"""Execution and reporting for planned cross-device tasks.

The runner is the only module that launches evaluation subprocesses. It also
writes generated configs and run manifests for both completed and skipped tasks.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from latency_bench.latency.runtime_telemetry import RuntimeTelemetryMonitor
from latency_bench.latency.profile import attach_hardware_fingerprint
from latency_bench.utils.io import read_json, write_json
from latency_bench.cross_device.configs import (
    _eval_command,
    _generated_config,
    _profile_command,
    _write_generated_config,
)
from latency_bench.cross_device.instances import INSTANCE_ID_ENV, resolve_instance_id, resolve_instance_origin
from latency_bench.cross_device.manifest import (
    append_manifest,
    append_note,
    build_base_manifest,
    completed_manifest_exists,
    host_fingerprint,
    now_iso,
    record_skipped_task,
)
from latency_bench.cross_device.planning import (
    PlannedTask,
    ProtocolTask,
    _build_plan,
    _output_root,
    _profiles_dir_for_args,
)
from latency_bench.cross_device.spec import detect_hardware
from latency_bench.cross_device.sync import hf_env, profile_remote_dir, upload_folder_to_hf


def failed_run_summary(task: ProtocolTask, exc: BaseException) -> dict[str, str]:
    return {
        "stage": str(task.stage),
        "policy": str(task.row["policy"]),
        "env": str(task.row["env"]),
        "output_dir": str(task.output_dir),
        "error": str(exc),
    }


def print_failed_summary(failures: list[dict[str, str]]) -> None:
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


def _stage_instance_metadata(stage: str, dry_run: bool) -> tuple[str | None, str | None]:
    if dry_run:
        if os.environ.get(INSTANCE_ID_ENV):
            return os.environ[INSTANCE_ID_ENV], "manual_override"
        return None, None
    uploads_raw_results = hf_env() is not None and not os.environ.get("CROSS_DEVICE_NO_RESULT_UPLOAD")
    if stage in {"measure", "sim_same"} or uploads_raw_results:
        return resolve_instance_id(host_fingerprint()), resolve_instance_origin()
    return None, None


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
    detected = detect_hardware(spec, args.hardware)
    profiles_dir = _profiles_dir_for_args(args, out)
    instance_id, instance_origin = _stage_instance_metadata(stage, bool(args.dry_run))
    plan = _build_plan(
        args,
        spec,
        stage,
        out,
        profiles_dir,
        detected["device"],
        instance_id=instance_id,
        instance_origin=instance_origin,
    )
    if args.max_runs is not None:
        plan = plan[: int(args.max_runs)]
    if args.dry_run:
        _print_dry_run_plan(stage, plan, spec)
        return
    if not plan:
        print("No protocol runs selected.")
        return
    failures: list[dict[str, str]] = []
    for item in plan:
        if not item.runnable:
            record_skipped_task(item.task, spec, detected, item.skip_reason or "skipped", out)
            print(f"skipping {item.task.row.get('policy')} {item.task.stage}: {item.skip_reason}")
            continue
        task = item.task
        try:
            print(
                "running "
                f"stage={task.stage} policy={task.row.get('policy')} env={task.row.get('env')} "
                f"output_dir={task.output_dir}"
            )
            _run_task(task, spec, detected, args, out)
        except Exception as exc:
            failures.append(failed_run_summary(task, exc))
            print(
                "failed "
                f"stage={task.stage} policy={task.row.get('policy')} env={task.row.get('env')} "
                f"output_dir={task.output_dir}: {exc}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                print_failed_summary(failures)
                raise
    if failures:
        print_failed_summary(failures)
        raise SystemExit(1)


def _run_task(
    task: ProtocolTask,
    spec: dict[str, Any],
    detected: dict[str, Any],
    args: argparse.Namespace,
    out: Path,
) -> None:
    """Execute one runnable task and record its manifest.

    Args:
        task: Concrete task from the planner.
        spec: Loaded protocol spec.
        detected: Hardware detection result for manifest metadata.
        args: Parsed CLI namespace with resume/dry-run/overwrite flags.
        out: Protocol output root used for the aggregate manifest stream.

    Side effects:
        Writes generated config, per-run manifest, global manifest JSONL, and
        launches the eval/profile subprocesses for real runs.
    """
    if args.resume and completed_manifest_exists(task.output_dir):
        print(f"resume: skipping completed run {task.output_dir}")
        return

    eval_cmd = _eval_command(task)
    config = _generated_config(task, spec)
    profile_cmd = None
    manifest = build_base_manifest(task, spec, detected, eval_cmd, profile_cmd)
    if task.stage == "measure":
        calibration_config_path = _calibration_generated_config_path(task)
        calibration_path = _calibration_output_path(task)
        calibration_cmd = _calibration_command(task, spec, calibration_config_path, calibration_path)
        _add_calibration_manifest_fields(manifest, calibration_config_path, calibration_path, calibration_cmd)
    task.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _write_generated_config(task.generated_config_path, config)
        if task.stage == "measure":
            _write_generated_config(calibration_config_path, config)
            profile_cmd = _profile_command(task, spec, args.overwrite, config)
        print(f"generated_config: {task.generated_config_path}")
        print(shlex.join(eval_cmd))
        if profile_cmd is not None:
            print(shlex.join(profile_cmd))
        return

    try:
        if task.stage == "measure":
            _write_generated_config(calibration_config_path, config)
            write_json(task.output_dir / "manifest.json", manifest)
            subprocess.run(calibration_cmd, check=True)
            _apply_env_fps_calibration(config, calibration_path)
            profile_cmd = _profile_command(task, spec, args.overwrite, config)
            manifest["profile_command"] = shlex.join(profile_cmd)
        _write_generated_config(task.generated_config_path, config)
        write_json(task.output_dir / "manifest.json", manifest)
        _run_with_runtime_telemetry(
            lambda: subprocess.run(eval_cmd, check=True),
            task.output_dir,
            manifest,
        )
        if profile_cmd is not None:
            subprocess.run(profile_cmd, check=True)
            _finalize_profile_artifacts(task, out, manifest)
        manifest["status"] = "completed"
        manifest["completed_at"] = now_iso()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["completed_at"] = now_iso()
        manifest["error"] = str(exc)
        manifest["notes"] = append_note(manifest.get("notes"), str(exc))
        write_json(task.output_dir / "manifest.json", manifest)
        append_manifest(out, manifest)
        raise
    write_json(task.output_dir / "manifest.json", manifest)
    append_manifest(out, manifest)
    _upload_run_results(task, out, manifest, detected, spec)


def _run_with_runtime_telemetry(run_fn, output_dir: Path, manifest: dict[str, Any]) -> None:
    monitor = RuntimeTelemetryMonitor(output_dir)
    manifest.update(monitor.start())
    try:
        run_fn()
    finally:
        try:
            manifest.update(monitor.stop())
        except Exception as exc:
            manifest["runtime_telemetry_stop_error"] = str(exc)


def _upload_run_results(
    task: ProtocolTask,
    out: Path,
    manifest: dict[str, Any],
    detected: dict[str, Any],
    spec: dict[str, Any],
) -> None:
    """Mirror this completed run's results to HF for hands-off aggregation.

    Uploads (full, including debug) this machine's slice of the protocol output
    to `raw/<device_id>/instances/<instance_id>/<runtime_alias>/cross_device/...`
    so an aggregating host can pull every machine's manifests + run dirs +
    profiles via `aggregate` auto-download. The download side filters debug
    artifacts back out.

    Gated on `HF_PROFILE_REPO` (reuses the profiles repo) and opt-out env
    `CROSS_DEVICE_NO_RESULT_UPLOAD`. Failures are non-fatal — the run is already
    on local disk — and recorded as a manifest note.

    Args:
        task: Completed task whose results should be published.
        out: Protocol output root (the local `results/cross_device`).
        manifest: Run manifest mutated in place with upload status.
        detected: Hardware detection result (for the device id segment).
        spec: Loaded protocol spec (for the runtime alias segment).
    """
    hf = hf_env()
    if hf is None or os.environ.get("CROSS_DEVICE_NO_RESULT_UPLOAD"):
        return
    device_id = detected["device"]["id"]
    if task.instance_id is None:
        raise RuntimeError("raw result upload requires task.instance_id")
    runtime_alias = spec["runtime_env"]["name"]
    prefix = f"raw/{device_id}/instances/{task.instance_id}/{runtime_alias}/cross_device"

    uploads: list[tuple[Path, str]] = []
    manifests_dir = out / "manifests"
    if manifests_dir.exists():
        uploads.append((manifests_dir, f"{prefix}/manifests"))
    if task.output_dir.exists():
        uploads.append((task.output_dir, f"{prefix}/{task.output_dir.relative_to(out).as_posix()}"))

    try:
        for local_dir, path_in_repo in uploads:
            upload_folder_to_hf(
                repo_id=hf["repo_id"],
                local_dir=local_dir,
                repo_type=hf["repo_type"],
                path_in_repo=path_in_repo,
            )
        manifest["results_uploaded"] = True
        manifest["results_repo"] = hf["repo_id"]
        manifest["results_path_prefix"] = prefix
        write_json(task.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest["results_uploaded"] = False
        manifest["notes"] = append_note(manifest.get("notes"), f"results upload failed: {exc}")


def _finalize_profile_artifacts(task: ProtocolTask, out: Path, manifest: dict[str, Any]) -> None:
    """Record hardware provenance and optionally publish a freshly built profile.

    Writes `hardware.json` into the profile folder, merges the fingerprint into
    `profile.json`, then uploads the folder to Hugging Face when `HF_PROFILE_REPO`
    is set. Any failure is recorded as a manifest note instead of aborting the
    measure run, since the profile artifacts are already on local disk.

    Args:
        task: Completed measure task whose profile was just built.
        out: Protocol output root (parent of the local `profiles` cache).
        manifest: Run manifest mutated in place with provenance/upload status.
    """
    profile_dir = task.profile_output_dir
    if profile_dir is None:
        return
    try:
        attach_hardware_fingerprint(profile_dir, host_fingerprint())
    except Exception as exc:
        manifest["notes"] = append_note(manifest.get("notes"), f"hardware fingerprint failed: {exc}")

    hf = hf_env()
    if hf is None:
        manifest["hf_uploaded"] = False
        return
    try:
        remote_dir = profile_remote_dir(profile_path=profile_dir / "profile.json", profiles_dir=out / "profiles")
        upload_folder_to_hf(
            repo_id=hf["repo_id"],
            local_dir=profile_dir,
            repo_type=hf["repo_type"],
            path_in_repo=remote_dir,
        )
        manifest["hf_uploaded"] = True
        manifest["hf_repo"] = hf["repo_id"]
        manifest["hf_path_in_repo"] = remote_dir
    except Exception as exc:
        manifest["hf_uploaded"] = False
        manifest["notes"] = append_note(manifest.get("notes"), f"hf upload failed: {exc}")


def _calibration_generated_config_path(task: ProtocolTask) -> Path:
    return task.generated_config_path.parent / f"{task.generated_config_path.stem}.calibration.yaml"


def _calibration_output_path(task: ProtocolTask) -> Path:
    return task.output_dir / "env_fps_calibration.json"


def _calibration_command(task: ProtocolTask, spec: dict[str, Any], config_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "latency_bench.experiments.calibrate_env_fps_from_saturated_latency",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--target-latency-frames",
        str(task.row["target_latency_frames"]),
        "--worker-utilization-cap",
        str(spec["env_fps_calibration"]["worker_utilization_cap"]),
    ]


def _add_calibration_manifest_fields(
    manifest: dict[str, Any],
    calibration_config_path: Path,
    calibration_path: Path,
    calibration_cmd: list[str],
) -> None:
    manifest["calibration_generated_config_path"] = str(calibration_config_path)
    manifest["calibration_output_path"] = str(calibration_path)
    manifest["calibration_command"] = shlex.join(calibration_cmd)


def _apply_env_fps_calibration(config: dict[str, Any], calibration_path: Path) -> None:
    payload = read_json(calibration_path)
    config["env"]["env_fps"] = float(payload["selected_env_fps"])
    config["env"]["obs_fps"] = float(payload["selected_obs_fps"])


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
        print(f"target_profile_id: {task.target_device.get('profile_id')}")
        print(f"target_profile_instance_id: {task.target_profile_instance_id or ''}")
        print(f"replay_device: {task.replay_device.get('id') if task.replay_device else ''}")
        print(f"smoke: {str(task.smoke).lower()}")
        print(f"episodes: {task.n_episodes}")
        print(f"status: {'runnable' if item.runnable else 'skipped'}")
        if item.skip_reason:
            print(f"skip_reason: {item.skip_reason}")
        if task.target_profile_path is not None:
            print(f"profile_path: {task.target_profile_path}")
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
            calibration_config_path = _calibration_generated_config_path(task)
            _write_generated_config(calibration_config_path, config)
            print(f"calibration_generated_config_path: {calibration_config_path}")
            print(f"calibration_output_path: {_calibration_output_path(task)}")
            print(
                "calibration_command: "
                f"{shlex.join(_calibration_command(task, spec, calibration_config_path, _calibration_output_path(task)))}"
            )
            if task.profile_output_dir is not None:
                print(f"profile_command: {shlex.join(_profile_command(task, spec, False, config))}")
        print(f"eval_max_steps: {config.get('evaluation', {}).get('eval_max_steps', '')}")
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
    if "replay hardware not configured" in reason:
        return "replay hardware not configured"
    if "missing profile" in reason:
        return "missing profile"
    if "checkpoint" in reason or "base_model_path" in reason:
        return "missing checkpoint"
    if "hardware mismatch" in reason:
        return "hardware mismatch"
    return "other"
