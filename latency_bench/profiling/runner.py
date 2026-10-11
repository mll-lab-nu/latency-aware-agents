"""Measure user-configured deployments and fit their recorded latency."""

from __future__ import annotations

import fcntl
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from latency_bench.eval.config import _evaluation_seed
from latency_bench.executors.env_step_backend import START_METHOD_ENV
from latency_bench.executors.thread_env import set_thread_env_defaults
from latency_bench.latency.hardware_info import (
    collect_hardware_fingerprint, describe_inference_deployment, public_hardware_projection,
)
from latency_bench.latency.profile_builder import build_profile_from_traces
from latency_bench.profiling.artifacts import profile_ref, simulation_seed_dir
from latency_bench.profiling.calibration import calibrate_env_fps_from_saturated_latency
from latency_bench.policy.model_source import resolve_model_source
from latency_bench.eval.artifacts import read_runtime_snapshot
from latency_bench.profiling.runtime_configs import (
    eval_command,
    evaluation_environment,
    measure_config,
    simulation_config,
)
from latency_bench.profiling.spec import path_segment, resolve_profile_config
from latency_bench.profiling.sync import publish_run_and_profile, publish_simulation_dirs
from latency_bench.utils.io import read_json, write_json, write_yaml


def run_profile(
    config: dict[str, Any], *, resume_run_id: str | None = None, publish: bool = False,
) -> dict[str, str]:
    """Measure and fit one deployment using the ordinary realtime evaluation path."""
    set_thread_env_defaults()
    config = resolve_profile_config(resolve_model_source(config))
    if config["executor"]["mode"] != "realtime":
        raise ValueError("Profiling requires executor.mode: realtime")
    options = config["profiling"]
    publish_repo = options["publish_repo"] if publish else None
    fingerprint = collect_hardware_fingerprint()
    deployment = describe_inference_deployment(fingerprint, config["executor"]["inference_devices"])
    ref = profile_ref(options["model_id"], deployment["gpu_class"], options["workload_id"], deployment["instance_id"])
    run_id = path_segment(resume_run_id) if resume_run_id is not None else _new_run_id()
    source_dir = Path(config["logging"]["output_dir"]) / "runs" / ref / run_id
    if resume_run_id is None:
        source_dir.mkdir(parents=True)
        write_json(source_dir / "hardware.json", fingerprint)
        write_json(source_dir / "run_contract.json", _run_contract(config, deployment))
    else:
        (source_dir / "hardware.json").read_bytes()
        _assert_run_contract(config, deployment, source_dir)

    calibration = _calibrate(config, deployment, source_dir)
    for session_id in range(options["repeats"]):
        _measure_session(config, deployment, calibration, source_dir, session_id,
                         resume=resume_run_id is not None)
    fit_profile(source_dir)
    current_dir = Path(config["logging"]["output_dir"]) / "profiles" / ref
    with _profile_lock(current_dir):
        _replace_current(source_dir / "profile", current_dir)
        if publish:
            publish_profile(source_dir, repo_id=publish_repo)
    return {
        "PROFILE_REF": ref.as_posix(), "PROFILE_PATH": str(current_dir / "profile.json"),
        "RUN_DIR": str(source_dir), "RUN_ID": run_id,
        "GPU_CLASS": deployment["gpu_class"], "INSTANCE_ID": deployment["instance_id"],
        "MEASUREMENT_STATUS": "complete", "FIT_STATUS": "complete",
        "PUBLICATION_STATUS": "published" if publish else "local",
    }


def fit_profile(source_dir: str | Path) -> dict[str, str]:
    """Fit completed local sessions without loading the model or remeasuring hardware."""
    source_dir = Path(source_dir).expanduser().absolute()
    contract = read_json(source_dir / "run_contract.json")
    config = read_runtime_snapshot(contract["config"])
    options = config["profiling"]
    calibration = read_json(source_dir / "calibration.json") if options["calibration"] is not None else None
    config = measure_config(config, contract["deployment"], source_dir, calibration)
    session_ids = list(range(options["repeats"]))
    for session_id in session_ids:
        (source_dir / "sessions" / str(session_id) / "DONE").read_bytes()
    candidate = source_dir / "profile"
    deployment = contract["deployment"]
    if not (candidate / "DONE").exists():
        if candidate.exists():
            shutil.rmtree(candidate)
        provenance = {
            key: {name: value for name, value in artifact.items() if name != "local_root"}
            for key, artifact in config["provenance"].items()
            if key in {"model_artifact", "backbone_artifact"}
        }
        build_profile_from_traces(
            sessions=[{"measure_session_id": session_id,
                       "observation_attempts": source_dir / "sessions" / str(session_id) / "observation_attempts.jsonl"}
                      for session_id in session_ids],
            output_dir=candidate, model_id=options["model_id"], gpu_class=deployment["gpu_class"],
            workload_id=options["workload_id"], instance_id=deployment["instance_id"],
            source_run_id=source_dir.name, env_fps=config["env"]["env_fps"],
            metadata={
                **provenance, "session_ids": session_ids,
                "measurement": {
                    "seed": _evaluation_seed(config), "evaluation": config["evaluation"],
                    "env_fps": config["env"]["env_fps"], "obs_fps": config["env"]["obs_fps"],
                    "inference_devices": config["executor"]["inference_devices"],
                    "policy_type": config["policy"]["type"],
                },
            },
        )
        hardware = public_hardware_projection(read_json(source_dir / "hardware.json"), deployment=deployment)
        hardware["instance_id"] = deployment["instance_id"]
        write_json(candidate / "hardware.json", hardware)
        (candidate / "DONE").touch()
    (source_dir / "DONE").touch()
    return {"PROFILE_PATH": str(candidate / "profile.json"), "RUN_DIR": str(source_dir), "FIT_STATUS": "complete"}


def publish_profile(source_dir: str | Path, *, repo_id: str) -> dict[str, str]:
    """Publish a completed local run by its artifact address, on any machine."""
    source_dir = Path(source_dir).expanduser().absolute()
    (source_dir / "DONE").read_bytes()
    candidate = source_dir / "profile"
    (candidate / "DONE").read_bytes()
    profile = read_json(candidate / "profile.json")
    ref = profile_ref(*(path_segment(profile[key]) for key in (
        "model_id", "gpu_class", "workload_id", "instance_id",
    )))
    publish_run_and_profile(repo_id, local_run_dir=source_dir, local_profile_dir=candidate,
                            run_ref=ref / path_segment(profile["source_run_id"]), profile_ref=ref)
    return {"PROFILE_REF": ref.as_posix(), "RUN_ID": profile["source_run_id"], "PUBLICATION_STATUS": "published"}


def run_simulation(
    config: dict[str, Any], *, profile_path: str | Path, method: str,
    latency_seeds: list[int], publish: bool = False,
) -> dict[str, str]:
    """Replay local profiles using their recorded artifact sources."""
    profile_path = Path(profile_path).expanduser().absolute()
    profile = read_json(profile_path)
    config = resolve_profile_config(resolve_model_source(config, profile=profile))
    options = config["profiling"]
    publish_repo = options["publish_repo"] if publish else None
    ref = profile_ref(*(path_segment(profile[key]) for key in (
        "model_id", "gpu_class", "workload_id", "instance_id",
    )))
    run_id = path_segment(profile["source_run_id"])
    output_dirs = [simulation_seed_dir(config["logging"]["output_dir"], ref, run_id, path_segment(method), seed)
                   for seed in latency_seeds]
    for output_dir in output_dirs:
        if (output_dir / "DONE").exists():
            raise FileExistsError(output_dir / "DONE")
    for seed, output_dir in zip(latency_seeds, output_dirs):
        if output_dir.exists():
            shutil.rmtree(output_dir)
        generated = output_dir / "generated.yaml"
        write_yaml(generated, simulation_config(
            config, output_dir=output_dir, profile_path=profile_path, profile=profile,
            method=method, latency_seed=seed,
        ))
        _run_evaluation(config, generated, output_dir)
        (output_dir / "DONE").touch()
    if publish:
        publish_simulation_dirs(publish_repo, local_dirs=output_dirs,
                                remote_dirs=[Path("simulations") / ref / run_id / method / str(seed)
                                             for seed in latency_seeds])
    return {"PROFILE_REF": ref.as_posix(), "RUN_ID": run_id, "SIMULATION_STATUS": "complete",
            "PUBLICATION_STATUS": "published" if publish else "local"}


def _calibrate(
    config: dict[str, Any], deployment: dict[str, Any], source_dir: Path,
) -> dict[str, Any] | None:
    options = config["profiling"]["calibration"]
    if options is None:
        return None
    calibration_path = source_dir / "calibration.json"
    if (source_dir / "calibration.DONE").exists():
        return read_json(calibration_path)
    generated = source_dir / "calibration.generated.yaml"
    write_yaml(generated, measure_config(config, deployment, source_dir / "calibration_probe"))
    calibration = calibrate_env_fps_from_saturated_latency(config_path=generated, output=calibration_path, **options)
    (source_dir / "calibration.DONE").touch()
    return calibration


def _run_contract(config: dict[str, Any], deployment: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": config,
        "deployment": deployment,
        "env_start_method": evaluation_environment()[START_METHOD_ENV],
        "evaluation_python": sys.executable,
    }


def _assert_run_contract(
    config: dict[str, Any], deployment: dict[str, Any], source_dir: Path,
) -> None:
    path = source_dir / "run_contract.json"
    actual = read_json(path)
    expected = deepcopy(_run_contract(config, deployment))
    # Publication targets do not change the measurement protocol.
    actual["config"]["profiling"].pop("publish_repo", None)
    expected["config"]["profiling"].pop("publish_repo", None)
    if actual != expected:
        raise ValueError(f"Profile run contract does not match: {path}")


def _measure_session(
    config: dict[str, Any], deployment: dict[str, Any], calibration: dict[str, Any] | None,
    source_dir: Path, session_id: int, *, resume: bool,
) -> None:
    session = source_dir / "sessions" / str(session_id)
    done = session / "DONE"
    if done.exists():
        if resume:
            return
        raise FileExistsError(done)
    if session.exists():
        shutil.rmtree(session)
    generated = session / "generated.yaml"
    write_yaml(generated, measure_config(config, deployment, session, calibration))
    _run_evaluation(config, generated, session)
    done.touch()


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


def _run_evaluation(config: dict[str, Any], config_path: Path, output_dir: Path) -> None:
    subprocess.run(eval_command(config["profiling"], config_path, output_dir),
                   env=evaluation_environment(), check=True)


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
