"""Execution and reporting for planned cross-device tasks.

The runner is the only module that launches evaluation subprocesses. It also
writes generated configs and run manifests for both completed and skipped tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    git_sha,
    host_fingerprint,
    now_iso,
    record_skipped_task,
)
from latency_bench.cross_device.planning import (
    STAGE_TO_EXPERIMENTS,
    PlannedTask,
    ProtocolTask,
    _build_plan,
    _make_task,
    _output_root,
    _profiles_dir_for_args,
)
from latency_bench.cross_device.spec import detect_hardware, slugify
from latency_bench.cross_device.sync import (
    download_measure_completions,
    download_profile_build,
    hf_env,
    profile_remote_dir,
    protocol_raw_root,
    publish_protocol_completion,
    publish_protocol_folder,
    publish_protocol_paths,
    upload_folder_to_hf,
)


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


def _protocol_task(
    *,
    spec: dict[str, Any],
    row: dict[str, Any],
    target_device: dict[str, Any],
    target_instance_id: str,
    target_instance_origin: str,
    stage: str,
    output_root: Path,
    measure_session_id: int | None = None,
    measure_role: str | None = None,
    latency_seed: int | None = None,
    sim_host_device: dict[str, Any] | None = None,
    sim_host_instance_id: str | None = None,
    target_profile_path: Path | None = None,
    profile_build_id: str | None = None,
    latency_method: str | None = None,
) -> ProtocolTask:
    seed = spec["seed_set"][0]
    return _make_task(
        stage=stage,
        experiment_ids=list(STAGE_TO_EXPERIMENTS[stage]),
        row=row,
        target_device=target_device,
        replay_device=sim_host_device,
        seed=seed,
        seed_set=spec["seed_set"],
        latency_seed=latency_seed,
        latency_seed_set=spec["latency_seed_set"],
        n_episodes=spec["n_episodes"],
        checkpoint_kind=spec["checkpoint_kind"],
        out=output_root,
        target_profile_path=target_profile_path,
        profile_output_dir=None,
        instance_id=sim_host_instance_id if sim_host_device is not None else target_instance_id,
        instance_origin=resolve_instance_origin() if sim_host_device is not None else target_instance_origin,
        target_profile_instance_id=target_instance_id,
        measure_session_id=measure_session_id,
        measure_role=measure_role,
        profile_build_id=profile_build_id,
        latency_method=latency_method,
    )


def _protocol_row(spec: dict[str, Any]) -> dict[str, Any]:
    return spec["rows"][0]


def _calibrate_protocol_once(
    *,
    args: argparse.Namespace,
    spec: dict[str, Any],
    row: dict[str, Any],
    device: dict[str, Any],
    instance_id: str,
    instance_origin: str,
    output_root: Path,
) -> Path:
    calibration_dir = output_root / "calibration" / slugify(row["policy"]) / device["id"] / instance_id
    calibration_path = calibration_dir / "env_fps_calibration.json"
    if calibration_path.exists():
        print(f"calibration: reusing {calibration_path}")
        return calibration_path
    task = _protocol_task(
        spec=spec,
        row=row,
        target_device=device,
        target_instance_id=instance_id,
        target_instance_origin=instance_origin,
        stage="measure",
        output_root=output_root,
        measure_session_id=spec["measure_sessions"]["fit"][0],
        measure_role="fit",
    )
    config_path = calibration_dir / "config.yaml"
    _write_generated_config(config_path, _generated_config(task, spec))
    command = _calibration_command(task, spec, config_path, calibration_path)
    print("calibration: " + shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)
    return calibration_path


def _run_measure_sessions(
    *,
    args: argparse.Namespace,
    spec: dict[str, Any],
    row: dict[str, Any],
    detected: dict[str, Any],
    instance_id: str,
    instance_origin: str,
    output_root: Path,
    calibration_path: Path,
    role: str,
    session_ids: list[int],
    profile_build_id: str | None = None,
) -> list[ProtocolTask]:
    tasks = []
    for session_id in session_ids:
        task = _protocol_task(
            spec=spec,
            row=row,
            target_device=detected["device"],
            target_instance_id=instance_id,
            target_instance_origin=instance_origin,
            stage="measure",
            output_root=output_root,
            measure_session_id=session_id,
            measure_role=role,
            profile_build_id=profile_build_id,
        )
        print(f"measure: role={role} session={session_id} output={task.output_dir}")
        _run_task(
            task,
            spec,
            detected,
            args,
            output_root,
            shared_calibration_path=calibration_path,
        )
        tasks.append(task)
    return tasks


def _profile_build_id(
    spec: dict[str, Any],
    device: dict[str, Any],
    instance_id: str,
    calibration_path: Path,
    fit_tasks: list[ProtocolTask],
) -> str:
    digest = hashlib.sha256()
    payload = {
        "protocol_id": spec["protocol_id"],
        "target_device": device["id"],
        "instance_id": instance_id,
        "calibration": {
            key: read_json(calibration_path)[key]
            for key in ("selected_env_fps", "selected_obs_fps")
        },
        "fit_sessions": [task.measure_session_id for task in fit_tasks],
        "git_sha": git_sha(),
        "profile_fitter_sha256": _profile_fitter_sha256(),
    }
    digest.update(json.dumps(payload, sort_keys=True).encode())
    for task in fit_tasks:
        for filename in (
            "e2e_latencies.jsonl",
            "infer_latencies.jsonl",
            "episode_metrics.jsonl",
            "actions.jsonl",
            "steps.jsonl",
        ):
            digest.update(filename.encode())
            digest.update((task.output_dir / filename).read_bytes())
    return digest.hexdigest()[:12]


def _profile_fitter_sha256() -> str:
    from latency_bench.experiments import build_latency_profile_from_trace
    from latency_bench.latency import burst_profile

    digest = hashlib.sha256()
    for module in (build_latency_profile_from_trace, burst_profile):
        path = Path(module.__file__)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _profile_artifact_sha256(profile_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(profile_dir.iterdir()):
        if not path.is_file() or path.name == "profile.json" or path.suffix == ".png":
            continue
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fit_protocol_profile(
    *,
    spec: dict[str, Any],
    row: dict[str, Any],
    device: dict[str, Any],
    instance_id: str,
    instance_origin: str,
    output_root: Path,
    calibration_path: Path,
    fit_tasks: list[ProtocolTask],
) -> tuple[str, Path, str]:
    from latency_bench.experiments.build_latency_profile_from_trace import build_profile_from_traces

    build_id = _profile_build_id(spec, device, instance_id, calibration_path, fit_tasks)
    profile_dir = (
        output_root
        / "profile_builds"
        / row["model_type"]
        / row["env_name"]
        / device["id"]
        / instance_id
        / build_id
    )
    if (profile_dir / "profile.json").exists():
        print(f"profile: reusing {profile_dir}")
        profile = read_json(profile_dir / "profile.json")
        return build_id, profile_dir, profile["profile_artifact_sha256"]
    manifests = [read_json(task.output_dir / "manifest.json") for task in fit_tasks]
    sessions = [
        {
            "measure_session_id": task.measure_session_id,
            "latencies": task.output_dir / "e2e_latencies.jsonl",
            "inference_latencies": task.output_dir / "infer_latencies.jsonl",
            "episode_metrics": task.output_dir / "episode_metrics.jsonl",
            "actions": task.output_dir / "actions.jsonl",
            "steps": task.output_dir / "steps.jsonl",
            "provenance": {
                "run_dir": str(task.output_dir.relative_to(output_root)),
                **{
                    key: manifest[key]
                    for key in (
                        "protocol_id",
                        "measure_session_id",
                        "measure_role",
                        "seed",
                        "n_episodes",
                        "git_sha",
                        "started_at",
                        "completed_at",
                    )
                },
            },
        }
        for task, manifest in zip(fit_tasks, manifests)
    ]
    calibration = read_json(calibration_path)
    build_profile_from_traces(
        sessions=sessions,
        output_dir=profile_dir,
        profile_id=device["profile_id"],
        policy_id=row["policy"],
        env_id=row["env_name"],
        target_device=device["id"],
        env_fps=calibration["selected_env_fps"],
        metadata={
            "policy_type": row["policy_type"],
            "model_type": row["model_type"],
            "hardware_class": device["id"],
            "instance_id": instance_id,
            "instance_origin": instance_origin,
            "profile_build_id": build_id,
            "profile_fitter_sha256": _profile_fitter_sha256(),
            "hardware_fingerprint": manifests[0]["hardware_fingerprint"],
            "calibration": calibration,
        },
        latency_kind="e2e_latency",
    )
    profile_path = profile_dir / "profile.json"
    profile = read_json(profile_path)
    artifact_sha256 = _profile_artifact_sha256(profile_dir)
    profile["profile_artifact_sha256"] = artifact_sha256
    write_json(profile_path, profile)
    return build_id, profile_dir, artifact_sha256


def _measure_completion_path(
    output_root: Path, device_id: str, instance_id: str, profile_build_id: str
) -> Path:
    return output_root / "completion" / "measure" / device_id / instance_id / f"{profile_build_id}.json"


def run_multi_session_measure(args: argparse.Namespace, spec: dict[str, Any]) -> None:
    output_root = _output_root(spec, args.out)
    detected = detect_hardware(spec, args.hardware)
    device = detected["device"]
    if device["id"] not in spec["target_devices"]:
        raise SystemExit(f"this protocol does not measure target device {device['id']}")
    instance_id = resolve_instance_id(host_fingerprint())
    instance_origin = resolve_instance_origin()
    completed = sorted(
        (output_root / "completion" / "measure" / device["id"] / instance_id).glob("*.json")
    )
    if completed:
        for path in completed:
            print(f"measure already complete: {path}")
        return
    row = _protocol_row(spec)
    calibration_path = _calibrate_protocol_once(
        args=args,
        spec=spec,
        row=row,
        device=device,
        instance_id=instance_id,
        instance_origin=instance_origin,
        output_root=output_root,
    )
    fit_tasks = _run_measure_sessions(
        args=args,
        spec=spec,
        row=row,
        detected=detected,
        instance_id=instance_id,
        instance_origin=instance_origin,
        output_root=output_root,
        calibration_path=calibration_path,
        role="fit",
        session_ids=spec["measure_sessions"]["fit"],
    )
    if args.dry_run:
        _run_measure_sessions(
            args=args,
            spec=spec,
            row=row,
            detected=detected,
            instance_id=instance_id,
            instance_origin=instance_origin,
            output_root=output_root,
            calibration_path=calibration_path,
            role="heldout",
            session_ids=spec["measure_sessions"]["heldout"],
            profile_build_id="profile_build",
        )
        return
    build_id, profile_dir, profile_artifact_sha256 = _fit_protocol_profile(
        spec=spec,
        row=row,
        device=device,
        instance_id=instance_id,
        instance_origin=instance_origin,
        output_root=output_root,
        calibration_path=calibration_path,
        fit_tasks=fit_tasks,
    )
    heldout_tasks = _run_measure_sessions(
        args=args,
        spec=spec,
        row=row,
        detected=detected,
        instance_id=instance_id,
        instance_origin=instance_origin,
        output_root=output_root,
        calibration_path=calibration_path,
        role="heldout",
        session_ids=spec["measure_sessions"]["heldout"],
        profile_build_id=build_id,
    )
    profile_manifest = read_json(profile_dir / "profile.json")
    completion = {
        "protocol_id": spec["protocol_id"],
        "policy": row["policy"],
        "model_type": row["model_type"],
        "env_name": row["env_name"],
        "target_device_id": device["id"],
        "target_profile_instance_id": instance_id,
        "profile_build_id": build_id,
        "profile_artifact_sha256": profile_artifact_sha256,
        "profile_fitter_sha256": profile_manifest["profile_fitter_sha256"],
        "seed_set": spec["seed_set"],
        "n_episodes": spec["n_episodes"],
        "measure_sessions": spec["measure_sessions"],
        "profile_path": str((profile_dir / "profile.json").relative_to(output_root)),
        "calibration_path": str(calibration_path.relative_to(output_root)),
        "fit_run_dirs": [str(task.output_dir.relative_to(output_root)) for task in fit_tasks],
        "heldout_run_dirs": [str(task.output_dir.relative_to(output_root)) for task in heldout_tasks],
        "generated_config_dirs": sorted(
            {
                str(task.generated_config_path.parent.relative_to(output_root))
                for task in [*fit_tasks, *heldout_tasks]
            }
        ),
    }
    path = _measure_completion_path(output_root, device["id"], instance_id, build_id)
    write_json(path, completion)
    print(f"measure completion: {path}")


def publish_measure(args: argparse.Namespace, spec: dict[str, Any]) -> None:
    output_root = _output_root(spec, args.out)
    device = detect_hardware(spec, args.hardware)["device"]
    instance_id = resolve_instance_id(host_fingerprint())
    completion_paths = sorted(
        (output_root / "completion" / "measure" / device["id"] / instance_id).glob("*.json")
    )
    if not completion_paths:
        raise FileNotFoundError(
            f"no completed measure profile for {device['id']}/{instance_id} under {output_root}"
        )
    for completion_path in completion_paths:
        completion = read_json(completion_path)
        remote_root = protocol_raw_root(device["id"], instance_id, spec["protocol_id"])
        profile_dir = (output_root / completion["profile_path"]).parent
        profile = read_json(profile_dir / "profile.json")
        artifact_sha256 = _profile_artifact_sha256(profile_dir)
        if not (
            artifact_sha256
            == profile["profile_artifact_sha256"]
            == completion["profile_artifact_sha256"]
        ):
            raise RuntimeError(f"profile artifact hash mismatch: {profile_dir}")
        if profile["profile_fitter_sha256"] != completion["profile_fitter_sha256"]:
            raise RuntimeError(f"profile fitter hash mismatch: {profile_dir}")
        publish_protocol_paths(
            spec,
            output_root=output_root,
            remote_root=remote_root,
            relative_paths=[
                str(Path(completion["calibration_path"]).parent),
                *completion["fit_run_dirs"],
                *completion["heldout_run_dirs"],
            ],
        )
        publish_protocol_paths(
            spec,
            output_root=output_root,
            remote_root=remote_root,
            relative_paths=completion["generated_config_dirs"],
        )
        publish_protocol_folder(
            spec,
            local_dir=profile_dir,
            path_in_repo=(
                f"profile_builds/{completion['model_type']}/{completion['env_name']}/"
                f"{device['id']}/{instance_id}/{completion['profile_build_id']}"
            ),
        )
        publish_protocol_completion(
            spec,
            local_path=completion_path,
            path_in_repo=f"{remote_root}/{completion_path.relative_to(output_root).as_posix()}",
        )
        print(f"published measure build: {completion['profile_build_id']}")


def _sim_completion_path(
    output_root: Path,
    latency_method: str,
    host_device_id: str,
    host_instance_id: str,
    target_device_id: str,
    target_instance_id: str,
    profile_build_id: str,
) -> Path:
    return (
        output_root
        / "completion"
        / "sim"
        / latency_method
        / host_device_id
        / host_instance_id
        / target_device_id
        / target_instance_id
        / f"{profile_build_id}.json"
    )


def run_all_profile_simulations(args: argparse.Namespace, spec: dict[str, Any]) -> None:
    output_root = _output_root(spec, args.out)
    detected = detect_hardware(spec, args.hardware)
    host_device = detected["device"]
    if host_device["id"] != spec["sim_host_device"]:
        raise SystemExit(f"this protocol runs sim on {spec['sim_host_device']}, got {host_device['id']}")
    if args.latency_method not in spec["latency_methods"]:
        raise SystemExit(f"latency method {args.latency_method!r} is not enabled by this protocol")
    host_instance_id = resolve_instance_id(host_fingerprint())
    downloaded = download_measure_completions(spec, output_root / "_hf_protocol")
    completion_paths = sorted(
        downloaded.glob(
            f"raw/*/instances/*/cross_device/{spec['protocol_id']}/completion/measure/**/*.json"
        )
    )
    measure_completions = [read_json(path) for path in completion_paths]
    available_devices = {
        completion["target_device_id"]
        for completion in measure_completions
    }
    missing_devices = [
        device_id
        for device_id in spec["target_devices"]
        if device_id not in available_devices
    ]
    if missing_devices:
        raise FileNotFoundError(
            "missing completed measure profiles for target devices: "
            + ", ".join(missing_devices)
        )
    devices = {device["id"]: device for device in spec["devices"]}
    row = _protocol_row(spec)
    for measure_completion in measure_completions:
        if measure_completion["target_device_id"] not in spec["target_devices"]:
            continue
        target_device = devices[measure_completion["target_device_id"]]
        target_instance_id = measure_completion["target_profile_instance_id"]
        build_id = measure_completion["profile_build_id"]
        profile_path = download_profile_build(spec, measure_completion, output_root)
        profile = read_json(profile_path)
        artifact_sha256 = _profile_artifact_sha256(profile_path.parent)
        if not (
            artifact_sha256
            == profile["profile_artifact_sha256"]
            == measure_completion["profile_artifact_sha256"]
        ):
            raise RuntimeError(f"profile artifact hash mismatch: {profile_path}")
        if profile["profile_fitter_sha256"] != measure_completion["profile_fitter_sha256"]:
            raise RuntimeError(f"profile fitter hash mismatch: {profile_path}")
        tasks = []
        for latency_seed in spec["latency_seed_set"]:
            task = _protocol_task(
                spec=spec,
                row=row,
                target_device=target_device,
                target_instance_id=target_instance_id,
                target_instance_origin="measure_completion",
                stage="sim_cross",
                output_root=output_root,
                latency_seed=latency_seed,
                sim_host_device=host_device,
                sim_host_instance_id=host_instance_id,
                target_profile_path=profile_path,
                profile_build_id=build_id,
                latency_method=args.latency_method,
            )
            print(
                f"sim: method={args.latency_method} target={target_device['id']}/{target_instance_id} "
                f"build={build_id} latency_seed={latency_seed}"
            )
            _run_task(task, spec, detected, args, output_root)
            tasks.append(task)
        completion_path = _sim_completion_path(
            output_root,
            args.latency_method,
            host_device["id"],
            host_instance_id,
            target_device["id"],
            target_instance_id,
            build_id,
        )
        write_json(
            completion_path,
            {
                "protocol_id": spec["protocol_id"],
                "policy": row["policy"],
                "latency_method": args.latency_method,
                "sim_host_device_id": host_device["id"],
                "sim_host_instance_id": host_instance_id,
                "target_device_id": target_device["id"],
                "target_profile_instance_id": target_instance_id,
                "profile_build_id": build_id,
                "profile_artifact_sha256": measure_completion["profile_artifact_sha256"],
                "profile_fitter_sha256": measure_completion["profile_fitter_sha256"],
                "seed_set": spec["seed_set"],
                "n_episodes": spec["n_episodes"],
                "profile_path": str(profile_path.relative_to(output_root)),
                "latency_seed_set": spec["latency_seed_set"],
                "run_dirs": [str(task.output_dir.relative_to(output_root)) for task in tasks],
                "generated_config_dirs": sorted(
                    {
                        str(task.generated_config_path.parent.relative_to(output_root))
                        for task in tasks
                    }
                ),
            },
        )
        remote_root = protocol_raw_root(host_device["id"], host_instance_id, spec["protocol_id"])
        publish_protocol_paths(
            spec,
            output_root=output_root,
            remote_root=remote_root,
            relative_paths=[
                *sorted({str(task.output_dir.parent.relative_to(output_root)) for task in tasks}),
                *sorted({str(task.generated_config_path.parent.relative_to(output_root)) for task in tasks}),
            ],
        )
        publish_protocol_completion(
            spec,
            local_path=completion_path,
            path_in_repo=f"{remote_root}/{completion_path.relative_to(output_root).as_posix()}",
        )
        print(f"published sim completion: {completion_path}")


def _run_task(
    task: ProtocolTask,
    spec: dict[str, Any],
    detected: dict[str, Any],
    args: argparse.Namespace,
    out: Path,
    *,
    shared_calibration_path: Path | None = None,
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
        if shared_calibration_path is None:
            calibration_config_path = _calibration_generated_config_path(task)
            calibration_path = _calibration_output_path(task)
            calibration_cmd = _calibration_command(task, spec, calibration_config_path, calibration_path)
            _add_calibration_manifest_fields(manifest, calibration_config_path, calibration_path, calibration_cmd)
        else:
            calibration_path = Path(shared_calibration_path)
            manifest["calibration_output_path"] = str(calibration_path)
            manifest["calibration_command"] = ""
    task.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _write_generated_config(task.generated_config_path, config)
        if task.stage == "measure":
            if shared_calibration_path is None:
                _write_generated_config(calibration_config_path, config)
            elif calibration_path.exists():
                _apply_env_fps_calibration(config, calibration_path)
            if task.measure_session_id is None:
                profile_cmd = _profile_command(task, spec, args.overwrite, config)
        print(f"generated_config: {task.generated_config_path}")
        print(shlex.join(eval_cmd))
        if profile_cmd is not None:
            print(shlex.join(profile_cmd))
        return

    try:
        if task.stage == "measure":
            write_json(task.output_dir / "manifest.json", manifest)
            if shared_calibration_path is None:
                _write_generated_config(calibration_config_path, config)
                subprocess.run(calibration_cmd, check=True)
            _apply_env_fps_calibration(config, calibration_path)
            if task.measure_session_id is None:
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
    if task.measure_session_id is None and task.profile_build_id is None:
        _upload_run_results(task, out, manifest, detected)


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
) -> None:
    """Mirror this completed run's results to HF for hands-off aggregation.

    Uploads (full, including debug) this machine's slice of the protocol output
    to `raw/<device_id>/instances/<instance_id>/cross_device/...`
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
    """
    hf = hf_env()
    if hf is None or os.environ.get("CROSS_DEVICE_NO_RESULT_UPLOAD"):
        return
    device_id = detected["device"]["id"]
    if task.instance_id is None:
        raise RuntimeError("raw result upload requires task.instance_id")
    prefix = f"raw/{device_id}/instances/{task.instance_id}/cross_device"

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
