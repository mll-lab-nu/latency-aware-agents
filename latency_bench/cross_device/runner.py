from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from latency_bench.cross_device.instances import resolve_instance_id
from latency_bench.cross_device.spec import detect_hardware
from latency_bench.cross_device.study_configs import (
    calibration_config,
    eval_command,
    measure_config,
    simulation_config,
    write_generated_config,
)
from latency_bench.cross_device.sync import download_measurements, publish_study_unit
from latency_bench.experiments.build_latency_profile_from_trace import build_profile_from_traces
from latency_bench.experiments.calibrate_env_fps_from_saturated_latency import (
    calibrate_env_fps_from_saturated_latency,
)
from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.latency.profile import attach_hardware_fingerprint
from latency_bench.utils.io import read_json


def run_measure(study: dict[str, Any], hardware: str, *, resume: bool = False) -> None:
    device = detect_hardware(study, hardware)["device"]
    fingerprint = collect_hardware_fingerprint()
    instance_id = resolve_instance_id(fingerprint)
    output_root = Path(study["output_root"])
    measure_dir = output_root / "measure" / device["id"] / instance_id
    if resume and (measure_dir / "DONE").exists():
        return

    calibration_config_path = measure_dir / "calibration.generated.yaml"
    calibration_path = measure_dir / "calibration.json"
    calibration_done = measure_dir / "calibration.DONE"
    if resume and calibration_done.exists():
        calibration = read_json(calibration_path)
    else:
        write_generated_config(
            calibration_config_path,
            calibration_config(study, device, instance_id, measure_dir / "calibration_probe"),
        )
        calibration = calibrate_env_fps_from_saturated_latency(
            config_path=calibration_config_path,
            output=calibration_path,
            target_latency_frames=study["policy"]["target_latency_frames"],
            worker_utilization_cap=study["calibration"]["worker_utilization_cap"],
        )
        calibration_done.touch()

    for session_id in _measurement_session_ids(study):
        session_dir = measure_dir / "sessions" / str(session_id)
        session_done = session_dir / "DONE"
        if resume and session_done.exists():
            continue
        if resume and session_dir.exists():
            shutil.rmtree(session_dir)
        config_path = session_dir / "generated.yaml"
        config = measure_config(
            study,
            device,
            instance_id,
            session_dir,
            calibration,
        )
        write_generated_config(config_path, config)
        _run_evaluation(study, config_path, session_dir)
        session_done.touch()

    profile_dir = measure_dir / "profile"
    profile_done = profile_dir / "DONE"
    if not (resume and profile_done.exists()):
        if resume and profile_dir.exists():
            shutil.rmtree(profile_dir)
        build_profile_from_traces(
            sessions=[
                _profile_session(measure_dir, session_id)
                for session_id in study["sessions"]["fit"]
            ],
            output_dir=profile_dir,
            profile_id=device["profile_id"],
            policy_id=study["policy"]["id"],
            env_id=study["policy"]["env_name"],
            target_device=device["id"],
            env_fps=calibration["selected_env_fps"],
            metadata={
                "study_id": study["study_id"],
                "policy_type": study["policy"]["policy_type"],
                "model_type": study["policy"]["model_type"],
                "instance_id": instance_id,
                "fit_session_ids": study["sessions"]["fit"],
                "evaluation_session_ids": study["sessions"]["evaluation"],
            },
            latency_kind="e2e_latency",
        )
        attach_hardware_fingerprint(profile_dir, fingerprint)
        profile_done.touch()
    (measure_dir / "DONE").touch()


def run_simulation(study: dict[str, Any], method: str, *, resume: bool = False) -> None:
    _require_method(study, method)
    measurements = download_measurements(study)
    output_root = Path(study["output_root"])
    devices = {device["id"]: device for device in study["devices"]}
    done_paths = sorted((measurements / "measure").glob("*/*/DONE"))
    if not done_paths:
        raise FileNotFoundError("no completed measurements")
    for done_path in done_paths:
        device = devices[done_path.parent.parent.name]
        instance_id = done_path.parent.name
        profile_path = done_path.parent / "profile" / "profile.json"
        profile = read_json(profile_path)
        method_dir = output_root / "sim" / device["id"] / instance_id / method
        if resume and (method_dir / "DONE").exists():
            continue
        for latency_seed in study["latency_seeds"]:
            run_dir = method_dir / str(latency_seed)
            run_done = run_dir / "DONE"
            if resume and run_done.exists():
                continue
            if resume and run_dir.exists():
                shutil.rmtree(run_dir)
            config_path = run_dir / "generated.yaml"
            config = simulation_config(
                study,
                device,
                instance_id,
                run_dir,
                profile_path,
                profile,
                method,
                latency_seed,
            )
            write_generated_config(config_path, config)
            _run_evaluation(study, config_path, run_dir)
            run_done.touch()
        (method_dir / "DONE").touch()


def publish_measure(study: dict[str, Any], hardware: str) -> None:
    device = detect_hardware(study, hardware)["device"]
    instance_id = resolve_instance_id(collect_hardware_fingerprint())
    output_root = Path(study["output_root"])
    measure_dir = output_root / "measure" / device["id"] / instance_id
    (measure_dir / "DONE").read_bytes()
    publish_study_unit(study, measure_dir, measure_dir.relative_to(output_root))


def publish_simulation(study: dict[str, Any], method: str) -> None:
    _require_method(study, method)
    output_root = Path(study["output_root"])
    done_paths = sorted((output_root / "sim").glob(f"*/*/{method}/DONE"))
    if not done_paths:
        raise FileNotFoundError(f"no completed {method} simulations")
    for done_path in done_paths:
        method_dir = done_path.parent
        publish_study_unit(study, method_dir, method_dir.relative_to(output_root))


def _run_evaluation(study: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    subprocess.run(eval_command(study, config_path, output_dir), check=True)


def _measurement_session_ids(study: dict[str, Any]) -> list[int]:
    return list(
        dict.fromkeys(
            [
                *study["sessions"]["fit"],
                *study["sessions"]["evaluation"],
            ]
        )
    )


def _profile_session(measure_dir: Path, session_id: int) -> dict[str, Any]:
    session_dir = measure_dir / "sessions" / str(session_id)
    return {
        "measure_session_id": session_id,
        "latencies": session_dir / "e2e_latencies.jsonl",
        "inference_latencies": session_dir / "infer_latencies.jsonl",
        "episode_metrics": session_dir / "episode_metrics.jsonl",
        "actions": session_dir / "actions.jsonl",
        "steps": session_dir / "steps.jsonl",
    }


def _require_method(study: dict[str, Any], method: str) -> None:
    if method not in study["latency_methods"]:
        raise ValueError(f"latency method is not part of this study: {method}")
