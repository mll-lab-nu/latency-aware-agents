"""Manifest construction and reporting for cross-device protocol runs."""
from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from latency_bench.cross_device.configs import _eval_command
from latency_bench.cross_device.models import ProtocolTask, RuntimeSelection
from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.utils.io import write_json


_HOST_FINGERPRINT: dict[str, Any] | None = None


def host_fingerprint() -> dict[str, Any]:
    global _HOST_FINGERPRINT
    if _HOST_FINGERPRINT is None:
        _HOST_FINGERPRINT = collect_hardware_fingerprint()
    return _HOST_FINGERPRINT


def runtime_process_metadata() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def build_base_manifest(
    task: ProtocolTask,
    spec: dict[str, Any],
    detected: dict[str, Any],
    runtime: RuntimeSelection,
    eval_cmd: list[str],
    profile_cmd: list[str] | None,
) -> dict[str, Any]:
    replay_device = task.replay_device
    gpu_uuid_hash = None
    if task.instance_origin == "gpu_uuid_hash":
        gpu_uuid_hash = task.instance_id.removeprefix("instance_")
    return {
        "protocol_id": spec["protocol_id"],
        "stage": task.stage,
        "experiment_id": task.experiment_id,
        "experiment_ids": task.experiment_ids,
        "panel": task.panel,
        "policy": task.row["policy"],
        "env": task.row["env"],
        "policy_type": task.row["policy_type"],
        "model_type": task.row["model_type"],
        "env_name": task.row["env_name"],
        "target_device_id": task.target_device["id"],
        "target_device_display_name": task.target_device["display_name"],
        "target_profile_id": task.target_device["profile_id"],
        "instance_id": task.instance_id,
        "instance_origin": task.instance_origin,
        "target_profile_instance_id": task.target_profile_instance_id,
        "gpu_uuid_hash": gpu_uuid_hash,
        "replay_device_id": None if replay_device is None else replay_device["id"],
        "replay_device_display_name": None if replay_device is None else replay_device["display_name"],
        "hardware_detected_raw": detected["raw"],
        "hardware_normalized": detected["device"]["id"],
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
        "smoke": task.smoke,
        "smoke_max_steps": task.smoke_max_steps,
        "seed_set": task.seed_set,
        "seed": task.seed,
        "status": "running",
        "skip_reason": "",
        "error": "",
        "started_at": now_iso(),
        "completed_at": None,
        "git_sha": git_sha(),
        "hostname": socket.gethostname(),
        "hardware_fingerprint": host_fingerprint(),
        **runtime_process_metadata(),
        "notes": "",
    }


def append_manifest(out: Path, manifest: dict[str, Any]) -> None:
    path = out / "manifests" / "runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, sort_keys=True) + "\n")


def record_skipped_task(
    task: ProtocolTask,
    spec: dict[str, Any],
    detected: dict[str, Any],
    runtime: RuntimeSelection,
    skip_reason: str,
    out: Path,
) -> None:
    manifest = build_base_manifest(task, spec, detected, runtime, _eval_command(task), None)
    manifest["status"] = "skipped"
    manifest["completed_at"] = now_iso()
    manifest["skip_reason"] = skip_reason
    manifest["notes"] = skip_reason
    task.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(task.output_dir / "manifest.json", manifest)
    append_manifest(out, manifest)


def completed_manifest_exists(output_dir: Path) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["status"] == "completed"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha() -> str | None:
    try:
        process = subprocess.Popen(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return None
    stdout, _ = process.communicate()
    if process.returncode != 0:
        return None
    return stdout.strip()


def append_note(existing: Any, note: str) -> str:
    if existing:
        return f"{existing}; {note}"
    return note


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
