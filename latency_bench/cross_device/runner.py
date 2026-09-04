from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from latency_bench.cross_device.artifacts import (
    profile_dir,
    profile_ref,
    run_dir,
    run_ref,
    run_session_dir,
    simulation_seed_dir,
)
from latency_bench.cross_device.instances import instance_id_from_gpu_deployment
from latency_bench.cross_device.profile_resolver import resolve_remote_profile
from latency_bench.cross_device.runtime_configs import (
    calibration_config,
    eval_command,
    measure_config,
    simulation_config,
    write_generated_config,
)
from latency_bench.cross_device.spec import gpu_name_registry
from latency_bench.cross_device.sync import (
    publish_run_and_profile,
    publish_simulation_dirs,
)
from latency_bench.experiments.calibrate_env_fps_from_saturated_latency import (
    calibrate_env_fps_from_saturated_latency,
)
from latency_bench.latency.hardware_info import (
    collect_hardware_fingerprint,
    describe_participating_gpus,
    public_hardware_projection,
)
from latency_bench.latency.profile_builder import build_profile_from_traces
from latency_bench.utils.io import read_json, write_json


_CONTRACT_OUTPUT_DIR = Path("__latency_bench_contract_output__")


def current_deployment(spec: dict[str, Any]) -> dict[str, Any]:
    deployment, _fingerprint = _collect_deployment(spec)
    return deployment


def run_profile(
    spec: dict[str, Any],
    *,
    resume_run_id: str | None,
    publish: bool,
) -> dict[str, str]:
    deployment, fingerprint = _collect_deployment(spec)
    current_ref = profile_ref(
        spec["model_id"],
        deployment["gpu_class"],
        spec["workload_id"],
        deployment["instance_id"],
    )
    run_id = resume_run_id or _new_run_id()
    source_ref = run_ref(current_ref, run_id)
    source_dir = run_dir(spec["artifact_root"], source_ref)
    if resume_run_id is None:
        source_dir.mkdir(parents=True)
        write_json(source_dir / "hardware.json", fingerprint)
        write_json(source_dir / "run_contract.json", _run_contract(spec, deployment))
    else:
        (source_dir / "hardware.json").read_bytes()
        _assert_run_contract(spec, deployment, source_dir)

    calibration = _calibrate(spec, deployment, source_dir)
    write_json(
        source_dir / "run_contract.json",
        _run_contract(spec, deployment, calibration),
    )
    session_ids = list(range(spec["session_count"]))
    for session_id in session_ids:
        _measure_session(
            spec,
            deployment,
            calibration,
            source_ref,
            session_id,
            resume=resume_run_id is not None,
        )

    candidate_dir = source_dir / "profile"
    if not (candidate_dir / "DONE").exists():
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        build_profile_from_traces(
            sessions=[
                _profile_session(
                    run_session_dir(spec["artifact_root"], source_ref, session_id),
                    session_id,
                )
                for session_id in session_ids
            ],
            output_dir=candidate_dir,
            model_id=spec["model_id"],
            gpu_class=deployment["gpu_class"],
            workload_id=spec["workload_id"],
            instance_id=deployment["instance_id"],
            source_run_id=run_id,
            env_fps=calibration["selected_env_fps"],
            metadata={
                "base_config": spec["base_config"],
                "checkpoint_kind": spec["checkpoint_kind"],
                "session_ids": session_ids,
                **(
                    {"model_artifact": spec["model_provenance"]}
                    if "model_provenance" in spec
                    else {"checkpoint_path": spec["checkpoint_path"]}
                ),
                **(
                    {"runtime_artifact": spec["runtime_provenance"]}
                    if "runtime_provenance" in spec
                    else {}
                ),
            },
        )
        public_hardware = public_hardware_projection(
            fingerprint,
            gpu_name_registry=gpu_name_registry(spec["gpu_models"]),
            cuda_visible_devices=_inference_cuda_visible_devices(spec),
        )
        public_hardware["instance_id"] = deployment["instance_id"]
        write_json(candidate_dir / "hardware.json", public_hardware)
        (candidate_dir / "DONE").touch()

    (source_dir / "DONE").touch()
    current_dir = profile_dir(spec["artifact_root"], current_ref)
    with _profile_lock(current_dir):
        _replace_current(candidate_dir, current_dir)
        published_commit = _publish(spec, current_ref, current_dir) if publish else None
    result = {
        "PROFILE_REF": current_ref.as_posix(),
        "PROFILE_PATH": str(current_dir / "profile.json"),
        "RUN_ID": run_id,
        "GPU_CLASS": deployment["gpu_class"],
        "INSTANCE_ID": deployment["instance_id"],
    }
    if published_commit is not None:
        result["HF_COMMIT"] = published_commit
    return result


def publish_current_profile(spec: dict[str, Any]) -> dict[str, str]:
    deployment = current_deployment(spec)
    current_ref = profile_ref(
        spec["model_id"],
        deployment["gpu_class"],
        spec["workload_id"],
        deployment["instance_id"],
    )
    current_dir = profile_dir(spec["artifact_root"], current_ref)
    with _profile_lock(current_dir):
        commit = _publish(spec, current_ref, current_dir)
        profile = read_json(current_dir / "profile.json")
    return {
        "PROFILE_REF": current_ref.as_posix(),
        "PROFILE_PATH": str(current_dir / "profile.json"),
        "RUN_ID": profile["source_run_id"],
        "HF_COMMIT": commit,
    }


def run_simulation(
    spec: dict[str, Any],
    *,
    gpu_class: str,
    instance_id: str,
    method: str,
    latency_seeds: list[int],
    hf_commit: str,
    publish: bool,
) -> dict[str, str]:
    canonical_ref, selected_profile, resolved_commit = resolve_remote_profile(
        spec,
        gpu_class=gpu_class,
        instance_id=instance_id,
        hf_commit=hf_commit,
    )
    profile = read_json(selected_profile)
    output_dirs = [
        simulation_seed_dir(
            spec["artifact_root"],
            canonical_ref,
            profile["source_run_id"],
            method,
            seed,
        )
        for seed in latency_seeds
    ]
    for output_dir in output_dirs:
        if (output_dir / "DONE").exists():
            raise FileExistsError(output_dir / "DONE")
    for latency_seed, output_dir in zip(latency_seeds, output_dirs):
        if output_dir.exists():
            shutil.rmtree(output_dir)
        config_path = output_dir / "generated.yaml"
        config = simulation_config(
            spec,
            output_dir=output_dir,
            profile_path=selected_profile,
            profile_ref=canonical_ref,
            profile=profile,
            method=method,
            latency_seed=latency_seed,
            hf_commit=resolved_commit,
        )
        write_generated_config(config_path, config)
        _run_evaluation(spec, config_path, output_dir)
        (output_dir / "DONE").touch()

    result = {
        "PROFILE_REF": canonical_ref.as_posix(),
        "RUN_ID": profile["source_run_id"],
        "HF_COMMIT": resolved_commit,
    }
    if publish:
        remote_dirs = [
            Path("simulations")
            / canonical_ref
            / profile["source_run_id"]
            / method
            / str(seed)
            for seed in latency_seeds
        ]
        result["SIMULATION_HF_COMMIT"] = publish_simulation_dirs(
            spec,
            local_dirs=output_dirs,
            remote_dirs=remote_dirs,
        )
    return result


def _collect_deployment(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fingerprint = collect_hardware_fingerprint()
    names = gpu_name_registry(spec["gpu_models"])
    deployment = describe_participating_gpus(
        gpus=fingerprint["gpus"],
        topology=fingerprint["gpu"]["topology"],
        gpu_name_registry=names,
        cuda_visible_devices=_inference_cuda_visible_devices(spec),
    )
    deployment["instance_id"] = instance_id_from_gpu_deployment(deployment)
    fingerprint["gpu_deployment"] = deployment
    return deployment, fingerprint


def _inference_cuda_visible_devices(spec: dict[str, Any]) -> str:
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    simulation_gpu = spec["simulation_gpu"]
    if simulation_gpu is None:
        return ",".join(visible)
    inference = [
        device
        for logical_index, device in enumerate(visible)
        if f"cuda:{logical_index}" != simulation_gpu
    ]
    if not inference:
        raise RuntimeError("GPU simulator profiling requires a separate inference GPU")
    return ",".join(inference)


def _calibrate(
    spec: dict[str, Any],
    deployment: dict[str, Any],
    source_dir: Path,
) -> dict[str, Any]:
    calibration_path = source_dir / "calibration.json"
    if (source_dir / "calibration.DONE").exists():
        return read_json(calibration_path)
    if not spec["calibrate_env_fps"]:
        base_config = calibration_config(spec, deployment, source_dir / "calibration_probe")
        calibration = {
            "selected_env_fps": base_config["env"]["env_fps"],
            "selected_obs_fps": base_config["env"]["obs_fps"],
            "calibration_source": "fixed_workload_cadence",
        }
        write_json(calibration_path, calibration)
        (source_dir / "calibration.DONE").touch()
        return calibration
    config_path = source_dir / "calibration.generated.yaml"
    write_generated_config(
        config_path,
        calibration_config(spec, deployment, source_dir / "calibration_probe"),
    )
    calibration = calibrate_env_fps_from_saturated_latency(
        config_path=config_path,
        output=calibration_path,
        target_latency_frames=spec["target_latency_frames"],
        worker_utilization_cap=spec["worker_utilization_cap"],
    )
    (source_dir / "calibration.DONE").touch()
    return calibration


def _run_contract(
    spec: dict[str, Any],
    deployment: dict[str, Any],
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = {
        "model_id": spec["model_id"],
        "workload_id": spec["workload_id"],
        "deployment": deployment,
        "checkpoint_kind": spec["checkpoint_kind"],
        "checkpoint_path": spec["checkpoint_path"],
        "runtime_checkpoint_path": (
            spec["runtime_checkpoint_path"]
            if "runtime_checkpoint_path" in spec
            else None
        ),
        "model_provenance": (
            spec["model_provenance"] if "model_provenance" in spec else None
        ),
        "runtime_provenance": (
            spec["runtime_provenance"] if "runtime_provenance" in spec else None
        ),
        "evaluation_command": eval_command(
            spec,
            _CONTRACT_OUTPUT_DIR / "generated.yaml",
            _CONTRACT_OUTPUT_DIR,
        ),
    }
    return {
        "identity": identity,
        "calibration_config": calibration_config(
            spec,
            deployment,
            _CONTRACT_OUTPUT_DIR / "calibration_probe",
        ),
        "measurement_config": (
            measure_config(
                spec,
                deployment,
                _CONTRACT_OUTPUT_DIR,
                calibration,
            )
            if calibration is not None
            else None
        ),
    }


def _assert_run_contract(
    spec: dict[str, Any],
    deployment: dict[str, Any],
    source_dir: Path,
) -> None:
    contract_path = source_dir / "run_contract.json"
    try:
        actual = read_json(contract_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Cannot resume run without local contract: {contract_path}"
        ) from exc

    calibration = (
        read_json(source_dir / "calibration.json")
        if (source_dir / "calibration.DONE").exists()
        else None
    )
    expected = _run_contract(spec, deployment, calibration)
    if actual != expected:
        raise ValueError(f"Profile run contract does not match: {contract_path}")


def _measure_session(
    spec: dict[str, Any],
    deployment: dict[str, Any],
    calibration: dict[str, Any],
    source_ref: Path,
    session_id: int,
    *,
    resume: bool,
) -> None:
    session_dir = run_session_dir(spec["artifact_root"], source_ref, session_id)
    done = session_dir / "DONE"
    if done.exists():
        if resume:
            return
        raise FileExistsError(done)
    if session_dir.exists():
        shutil.rmtree(session_dir)
    config_path = session_dir / "generated.yaml"
    write_generated_config(
        config_path,
        measure_config(spec, deployment, session_dir, calibration),
    )
    _run_evaluation(spec, config_path, session_dir)
    done.touch()


def _publish(
    spec: dict[str, Any],
    current_ref: Path,
    current_dir: Path,
) -> str:
    (current_dir / "DONE").read_bytes()
    profile = read_json(current_dir / "profile.json")
    source_ref = run_ref(current_ref, profile["source_run_id"])
    source_dir = run_dir(spec["artifact_root"], source_ref)
    (source_dir / "DONE").read_bytes()
    return publish_run_and_profile(
        spec,
        local_run_dir=source_dir,
        local_profile_dir=current_dir,
        run_ref=source_ref,
        profile_ref=current_ref,
    )


def _replace_current(source: Path, current: Path) -> None:
    current.parent.mkdir(parents=True, exist_ok=True)
    backup = current.with_name(f".{current.name}.old")
    if not current.exists() and backup.exists():
        backup.rename(current)
    elif backup.exists():
        shutil.rmtree(backup)
    replacement = Path(tempfile.mkdtemp(prefix=f".{current.name}.new-", dir=current.parent))
    shutil.rmtree(replacement)
    shutil.copytree(source, replacement)
    if current.exists():
        current.rename(backup)
    try:
        replacement.rename(current)
    except BaseException:
        if backup.exists():
            backup.rename(current)
        raise
    if backup.exists():
        shutil.rmtree(backup)


@contextmanager
def _profile_lock(current: Path):
    current.parent.mkdir(parents=True, exist_ok=True)
    with current.with_name(f".{current.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _run_evaluation(spec: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    subprocess.run(eval_command(spec, config_path, output_dir), check=True)


def _profile_session(session_dir: Path, session_id: int) -> dict[str, Any]:
    return {
        "measure_session_id": session_id,
        "observation_attempts": session_dir / "observation_attempts.jsonl",
    }


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
