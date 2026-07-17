from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from latency_bench.cross_device.artifacts import (
    profile_dir,
    profile_ref,
    raw_session_dir,
    raw_workload_dir,
    simulation_seed_dir,
    target_ref,
)
from latency_bench.cross_device.instances import resolve_instance_id
from latency_bench.cross_device.runtime_configs import (
    calibration_config,
    eval_command,
    measure_config,
    simulation_config,
    write_generated_config,
)
from latency_bench.cross_device.spec import detect_hardware
from latency_bench.cross_device.sync import (
    download_calibration,
    download_profile,
    download_raw_selection,
    publish_calibration,
    publish_directory_leaf,
)
from latency_bench.experiments.build_latency_profile_from_trace import build_profile_from_traces
from latency_bench.experiments.calibrate_env_fps_from_saturated_latency import (
    calibrate_env_fps_from_saturated_latency,
)
from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.latency.profile import attach_hardware_fingerprint
from latency_bench.utils.io import read_json, write_json


def current_target_ref(workload: dict[str, Any], hardware: str) -> Path:
    device, _, instance_id = _current_target(workload, hardware)
    return target_ref(device["id"], instance_id, workload["workload"]["id"])


def download_raw(
    workload: dict[str, Any],
    target: Path,
    session_ids: list[int],
    *,
    overwrite: bool = False,
) -> None:
    download_raw_selection(
        workload,
        target,
        raw_workload_dir(workload["artifact_root"], target),
        session_ids,
        overwrite=overwrite,
    )


def run_measure(
    workload: dict[str, Any],
    hardware: str,
    session_ids: list[int],
    *,
    resume: bool = False,
) -> Path:
    device, fingerprint, instance_id = _current_target(workload, hardware)
    target = target_ref(device["id"], instance_id, workload["workload"]["id"])
    artifact_root = Path(workload["artifact_root"])
    raw_dir = raw_workload_dir(artifact_root, target)
    calibration = _load_or_run_calibration(
        workload,
        device,
        instance_id,
        fingerprint,
        target,
        raw_dir,
    )
    for session_id in session_ids:
        session_dir = raw_session_dir(artifact_root, target, session_id)
        session_done = session_dir / "DONE"
        if session_done.exists():
            if resume:
                continue
            raise FileExistsError(session_done)
        if session_dir.exists():
            shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw_dir / "calibration.json", session_dir / "calibration.json")
        shutil.copyfile(raw_dir / "hardware.json", session_dir / "hardware.json")
        config_path = session_dir / "generated.yaml"
        config = measure_config(
            workload,
            device,
            instance_id,
            session_dir,
            calibration,
        )
        write_generated_config(config_path, config)
        _run_evaluation(workload, config_path, session_dir)
        session_done.touch()
    return target


def run_fit(
    workload: dict[str, Any],
    target: Path,
    profile_revision: str,
    session_ids: list[int],
    *,
    resume: bool = False,
) -> Path:
    device_id, instance_id, workload_id = target.parts
    if workload_id != workload["workload"]["id"]:
        raise ValueError(f"profile workload does not match config: {workload_id}")
    devices = {device["id"]: device for device in workload["devices"]}
    device = devices[device_id]
    artifact_root = Path(workload["artifact_root"])
    output_ref = profile_ref(target, profile_revision)
    output_dir = profile_dir(artifact_root, output_ref)
    output_done = output_dir / "DONE"
    if output_done.exists():
        if resume:
            return output_ref
        raise FileExistsError(output_done)
    raw_dir = raw_workload_dir(artifact_root, target)
    (raw_dir / "calibration.DONE").read_bytes()
    calibration = read_json(raw_dir / "calibration.json")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    sessions = []
    for session_id in session_ids:
        session_dir = raw_session_dir(artifact_root, target, session_id)
        (session_dir / "DONE").read_bytes()
        sessions.append(_profile_session(session_dir, session_id))
    build_profile_from_traces(
        sessions=sessions,
        output_dir=output_dir,
        profile_id=device["profile_id"],
        policy_id=workload_id,
        env_id=workload["workload"]["env_name"],
        target_device=device_id,
        env_fps=calibration["selected_env_fps"],
        metadata={
            "device_id": device_id,
            "instance_id": instance_id,
            "workload_id": workload_id,
            "profile_revision": profile_revision,
            "fit_session_ids": session_ids,
            "calibration_ref": (Path("raw") / target / "calibration.json").as_posix(),
            "policy_type": workload["workload"]["policy_type"],
            "model_type": workload["workload"]["model_type"],
            "measurement": workload["measurement"],
        },
        latency_kind="e2e_latency",
    )
    attach_hardware_fingerprint(output_dir, read_json(raw_dir / "hardware.json"))
    output_done.touch()
    return output_ref


def run_simulation(
    workload: dict[str, Any],
    canonical_profile_ref: Path,
    method: str,
    latency_seeds: list[int],
    *,
    resume: bool = False,
) -> None:
    device_id, _, workload_id, _ = canonical_profile_ref.parts
    if workload_id != workload["workload"]["id"]:
        raise ValueError(f"profile workload does not match config: {workload_id}")
    artifact_root = Path(workload["artifact_root"])
    pending = []
    for latency_seed in latency_seeds:
        output_dir = simulation_seed_dir(
            artifact_root,
            canonical_profile_ref,
            method,
            latency_seed,
        )
        output_done = output_dir / "DONE"
        if output_done.exists():
            if resume:
                continue
            raise FileExistsError(output_done)
        pending.append((latency_seed, output_dir))
    if not pending:
        return
    downloaded_profile = download_profile(workload, canonical_profile_ref)
    profile = read_json(downloaded_profile / "profile.json")
    for latency_seed, output_dir in pending:
        output_done = output_dir / "DONE"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        config_path = output_dir / "generated.yaml"
        config = simulation_config(
            workload,
            output_dir,
            downloaded_profile / "profile.json",
            canonical_profile_ref,
            profile,
            method,
            latency_seed,
        )
        write_generated_config(config_path, config)
        _run_evaluation(workload, config_path, output_dir)
        output_done.touch()


def publish_raw(
    workload: dict[str, Any],
    hardware: str,
    session_ids: list[int],
    *,
    resume: bool = False,
) -> Path:
    target = current_target_ref(workload, hardware)
    artifact_root = Path(workload["artifact_root"])
    raw_dir = raw_workload_dir(artifact_root, target)
    remote_raw = Path("raw") / target
    publish_calibration(workload, raw_dir, remote_raw, resume=resume)
    for session_id in session_ids:
        publish_directory_leaf(
            workload,
            raw_session_dir(artifact_root, target, session_id),
            remote_raw / "sessions" / str(session_id),
            resume=resume,
        )
    return target


def publish_profile(
    workload: dict[str, Any],
    canonical_profile_ref: Path,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    publish_directory_leaf(
        workload,
        profile_dir(workload["artifact_root"], canonical_profile_ref),
        Path("profiles") / canonical_profile_ref,
        resume=resume,
        overwrite=overwrite,
    )


def publish_simulation(
    workload: dict[str, Any],
    canonical_profile_ref: Path,
    method: str,
    latency_seeds: list[int],
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> None:
    for latency_seed in latency_seeds:
        publish_directory_leaf(
            workload,
            simulation_seed_dir(
                workload["artifact_root"],
                canonical_profile_ref,
                method,
                latency_seed,
            ),
            Path("sim") / canonical_profile_ref / method / str(latency_seed),
            resume=resume,
            overwrite=overwrite,
        )


def _load_or_run_calibration(
    workload: dict[str, Any],
    device: dict[str, Any],
    instance_id: str,
    fingerprint: dict[str, Any],
    target: Path,
    raw_dir: Path,
) -> dict[str, Any]:
    calibration_path = raw_dir / "calibration.json"
    calibration_done = raw_dir / "calibration.DONE"
    if calibration_done.exists():
        return read_json(calibration_path)
    _clear_incomplete_calibration(raw_dir)
    if download_calibration(workload, target, raw_dir):
        return read_json(calibration_path)
    config_path = raw_dir / "calibration.generated.yaml"
    write_generated_config(
        config_path,
        calibration_config(
            workload,
            device,
            instance_id,
            raw_dir / "calibration_probe",
        ),
    )
    calibration = calibrate_env_fps_from_saturated_latency(
        config_path=config_path,
        output=calibration_path,
        target_latency_frames=workload["workload"]["target_latency_frames"],
        worker_utilization_cap=workload["calibration"]["worker_utilization_cap"],
    )
    write_json(raw_dir / "hardware.json", fingerprint)
    calibration_done.touch()
    return calibration


def _clear_incomplete_calibration(raw_dir: Path) -> None:
    probe_dir = raw_dir / "calibration_probe"
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    for name in (
        "calibration.generated.yaml",
        "calibration.json",
        "hardware.json",
    ):
        path = raw_dir / name
        if path.exists():
            path.unlink()


def _current_target(
    workload: dict[str, Any],
    hardware: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    device = detect_hardware(workload, hardware)["device"]
    fingerprint = collect_hardware_fingerprint()
    return device, fingerprint, resolve_instance_id(fingerprint)


def _run_evaluation(
    workload: dict[str, Any],
    config_path: Path,
    output_dir: Path,
) -> None:
    subprocess.run(eval_command(workload, config_path, output_dir), check=True)


def _profile_session(session_dir: Path, session_id: int) -> dict[str, Any]:
    return {
        "measure_session_id": session_id,
        "latencies": session_dir / "e2e_latencies.jsonl",
        "inference_latencies": session_dir / "infer_latencies.jsonl",
        "episode_metrics": session_dir / "episode_metrics.jsonl",
        "actions": session_dir / "actions.jsonl",
        "steps": session_dir / "steps.jsonl",
    }
