"""Execution and reporting for planned cross-device tasks.

The runner is the only module that launches evaluation subprocesses. It also
writes generated configs and run manifests for both completed and skipped tasks.
"""
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.latency.profile import (
    load_latency_samples,
    load_profile_manifest,
    save_profile_manifest,
    write_latency_samples,
)
from scripts.cross_device.config import (
    _eval_command,
    _generated_config,
    _profile_command,
    _write_generated_config,
)
from scripts.cross_device.models import PlannedTask, ProtocolTask, RuntimeSelection
from scripts.cross_device.planning import _build_plan, _output_root, _profiles_dir_for_args
from scripts.cross_device.profile_cache import profile_remote_dir, upload_folder_to_hf
from scripts.cross_device.runtime import _stage_hardware, _validate_runtime_safety, detect_runtime_env


_HOST_FINGERPRINT: dict[str, Any] | None = None


def _host_fingerprint() -> dict[str, Any]:
    """Return this process's hardware fingerprint, collected once and cached.

    Returns:
        The fingerprint dict from `collect_hardware_fingerprint`; reused across
        every manifest in the run so the probes (nvidia-smi/lscpu) run once.
    """
    global _HOST_FINGERPRINT
    if _HOST_FINGERPRINT is None:
        _HOST_FINGERPRINT = collect_hardware_fingerprint()
    return _HOST_FINGERPRINT


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
    config = _generated_config(task, spec)
    profile_cmd = None
    manifest = _base_manifest(task, spec, detected, runtime, eval_cmd, profile_cmd)
    if task.stage == "measure":
        warmup_config_path = _warmup_generated_config_path(task)
        calibration_path = _calibration_output_path(task)
        calibration_cmd = _calibration_command(task, warmup_config_path, calibration_path)
        _add_calibration_manifest_fields(manifest, warmup_config_path, calibration_path, calibration_cmd)
    task.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _write_generated_config(task.generated_config_path, config)
        if task.stage == "measure":
            _write_generated_config(warmup_config_path, config)
            profile_cmd = _profile_command(task, spec, args.overwrite, config)
        print(f"generated_config: {task.generated_config_path}")
        print(shlex.join(eval_cmd))
        if profile_cmd is not None:
            print(shlex.join(profile_cmd))
        return

    try:
        if task.stage == "measure":
            _write_generated_config(warmup_config_path, config)
            _write_json(task.output_dir / "manifest.json", manifest)
            subprocess.run(calibration_cmd, check=True)
            _apply_env_fps_calibration(config, calibration_path)
            profile_cmd = _profile_command(task, spec, args.overwrite, config)
            manifest["profile_command"] = shlex.join(profile_cmd)
        _write_generated_config(task.generated_config_path, config)
        _write_json(task.output_dir / "manifest.json", manifest)
        if task.stage == "measure" and task.n_samples is not None:
            _run_measure_until_episode_and_sample_targets(task, config)
        else:
            subprocess.run(eval_cmd, check=True)
        if profile_cmd is not None:
            subprocess.run(profile_cmd, check=True)
            _finalize_profile_artifacts(task, out, manifest)
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
    _upload_run_results(task, out, manifest, detected, runtime)


def _upload_run_results(
    task: ProtocolTask,
    out: Path,
    manifest: dict[str, Any],
    detected: dict[str, Any],
    runtime: RuntimeSelection,
) -> None:
    """Mirror this completed run's results to HF for hands-off aggregation.

    Uploads (full, including debug) this machine's slice of the protocol output
    to `raw/<device_id>/<runtime_alias>/cross_device/...` so an aggregating host
    can pull every machine's manifests + run dirs + profiles via `aggregate`
    auto-download. The download side filters debug artifacts back out.

    Gated on `HF_PROFILE_REPO` (reuses the profiles repo) and opt-out env
    `CROSS_DEVICE_NO_RESULT_UPLOAD`. Failures are non-fatal — the run is already
    on local disk — and recorded as a manifest note.

    Args:
        task: Completed task whose results should be published.
        out: Protocol output root (the local `results/cross_device`).
        manifest: Run manifest mutated in place with upload status.
        detected: Hardware detection result (for the device id segment).
        runtime: Runtime detection result (for the runtime alias segment).
    """
    repo_id = os.environ.get("HF_PROFILE_REPO")
    if not repo_id or os.environ.get("CROSS_DEVICE_NO_RESULT_UPLOAD"):
        return
    device_id = (detected.get("device") or {}).get("id") or "unknown_device"
    runtime_alias = runtime.matched_alias or runtime.selected_alias or "unknown_runtime"
    prefix = f"raw/{device_id}/{runtime_alias}/cross_device"
    repo_type = os.environ.get("HF_PROFILE_REPO_TYPE", "dataset")

    uploads: list[tuple[Path, str]] = []
    manifests_dir = out / "manifests"
    if manifests_dir.exists():
        uploads.append((manifests_dir, f"{prefix}/manifests"))
    if task.output_dir.exists():
        uploads.append((task.output_dir, f"{prefix}/{task.output_dir.relative_to(out).as_posix()}"))
    profile_dir = task.profile_output_dir
    if profile_dir is not None and profile_dir.exists():
        uploads.append((profile_dir, f"{prefix}/{profile_dir.relative_to(out).as_posix()}"))

    try:
        for local_dir, path_in_repo in uploads:
            upload_folder_to_hf(
                repo_id=repo_id,
                local_dir=local_dir,
                repo_type=repo_type,
                path_in_repo=path_in_repo,
            )
        manifest["results_uploaded"] = True
        manifest["results_repo"] = repo_id
        manifest["results_path_prefix"] = prefix
        _write_json(task.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest["results_uploaded"] = False
        manifest["notes"] = _append_note(manifest.get("notes"), f"results upload failed: {exc}")


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
        fingerprint = _host_fingerprint()
        _write_json(profile_dir / "hardware.json", fingerprint)
        profile_path = profile_dir / "profile.json"
        if profile_path.exists():
            profile = load_profile_manifest(profile_path)
            profile["hardware"] = fingerprint
            save_profile_manifest(profile, profile_path)
    except Exception as exc:
        manifest["notes"] = _append_note(manifest.get("notes"), f"hardware fingerprint failed: {exc}")

    repo_id = os.environ.get("HF_PROFILE_REPO")
    if not repo_id:
        manifest["hf_uploaded"] = False
        return
    try:
        remote_dir = profile_remote_dir(profile_path=profile_dir / "profile.json", profiles_dir=out / "profiles")
        upload_folder_to_hf(
            repo_id=repo_id,
            local_dir=profile_dir,
            repo_type=os.environ.get("HF_PROFILE_REPO_TYPE", "dataset"),
            path_in_repo=remote_dir,
        )
        manifest["hf_uploaded"] = True
        manifest["hf_repo"] = repo_id
        manifest["hf_path_in_repo"] = remote_dir
    except Exception as exc:
        manifest["hf_uploaded"] = False
        manifest["notes"] = _append_note(manifest.get("notes"), f"hf upload failed: {exc}")


def _warmup_generated_config_path(task: ProtocolTask) -> Path:
    return task.generated_config_path.parent / f"{task.generated_config_path.stem}.warmup.yaml"


def _calibration_output_path(task: ProtocolTask) -> Path:
    return task.output_dir / "env_fps_calibration.json"


def _calibration_command(task: ProtocolTask, config_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "latency_bench.experiments.calibrate_env_fps_from_warmup",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--target-latency-frames",
        str(task.row["target_latency_frames"]),
    ]


def _add_calibration_manifest_fields(
    manifest: dict[str, Any],
    warmup_config_path: Path,
    calibration_path: Path,
    calibration_cmd: list[str],
) -> None:
    manifest["warmup_generated_config_path"] = str(warmup_config_path)
    manifest["calibration_output_path"] = str(calibration_path)
    manifest["calibration_command"] = shlex.join(calibration_cmd)


def _apply_env_fps_calibration(config: dict[str, Any], calibration_path: Path) -> None:
    payload = _load_json(calibration_path)
    config["env"]["env_fps"] = float(payload["selected_env_fps"])
    config["env"]["obs_fps"] = float(payload["selected_obs_fps"])


def _run_measure_until_episode_and_sample_targets(task: ProtocolTask, config: dict[str, Any]) -> None:
    samples_ms: list[float] = []
    episode_records: list[dict[str, Any]] = []
    episode_index = 0
    while len(episode_records) < task.n_episodes or len(samples_ms) < task.n_samples:
        batch_output_dir = task.output_dir / "sample_batches" / f"episode_{episode_index}"
        batch_config_path = task.generated_config_path.parent / f"{task.generated_config_path.stem}.sample_{episode_index}.yaml"
        batch_config = deepcopy(config)
        batch_config["experiment"]["seed"] = task.seed + episode_index
        batch_config["evaluation"]["eval_episodes"] = 1
        batch_config["logging"]["output_dir"] = str(batch_output_dir)
        _write_generated_config(batch_config_path, batch_config)
        subprocess.run(_sample_batch_eval_command(task, batch_config_path, batch_output_dir), check=True)
        samples_needed = task.n_samples - len(samples_ms)
        if samples_needed > 0:
            samples_ms.extend(load_latency_samples(batch_output_dir / "latencies.jsonl")[:samples_needed])
        write_latency_samples(samples_ms, task.output_dir / "latencies.jsonl")
        episodes_needed = task.n_episodes - len(episode_records)
        if episodes_needed > 0:
            episode_records.extend(_load_jsonl_records(batch_output_dir / "episode_metrics.jsonl")[:episodes_needed])
        _write_jsonl_records(episode_records, task.output_dir / "episode_metrics.jsonl")
        episode_index += 1


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl_records(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _sample_batch_eval_command(task: ProtocolTask, config_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "latency_bench.run",
        "--config",
        str(config_path),
        "--checkpoint-kind",
        task.checkpoint_kind,
        "--output-dir",
        str(output_dir),
    ]


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
        "model_type": task.row.get("model_type"),
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
        "profile_path": str(
            task.target_profile_path or (task.profile_output_dir / "profile.json" if task.profile_output_dir else "")
        ),
        "output_dir": str(task.output_dir),
        "generated_config_path": str(task.generated_config_path),
        "command": shlex.join(eval_cmd),
        "profile_command": "" if profile_cmd is None else shlex.join(profile_cmd),
        "checkpoint_kind": task.checkpoint_kind,
        "n_episodes": task.n_episodes,
        "n_samples": task.n_samples,
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
        "hardware_fingerprint": _host_fingerprint(),
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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        if task.n_samples is not None:
            print(f"samples: {task.n_samples}")
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
            warmup_config_path = _warmup_generated_config_path(task)
            _write_generated_config(warmup_config_path, config)
            print(f"warmup_generated_config_path: {warmup_config_path}")
            print(f"calibration_output_path: {_calibration_output_path(task)}")
            print(
                "calibration_command: "
                f"{shlex.join(_calibration_command(task, warmup_config_path, _calibration_output_path(task)))}"
            )
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
    if "runtime_env mismatch" in reason or "runtime_env selection mismatch" in reason:
        return "runtime_env mismatch"
    if "replay hardware not configured" in reason:
        return "replay hardware not configured"
    if "missing profile" in reason:
        return "missing profile"
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
