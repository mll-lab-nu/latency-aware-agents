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


def load_protocol_manifests(root: str | Path, *, raw_root: str | Path | None = None) -> list[dict[str, Any]]:
    if raw_root is not None:
        return load_raw_run_manifests(raw_root)
    return load_run_manifests(root)


def load_run_manifests(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / "manifests" / "runs.jsonl"
    if not path.exists():
        return []
    records = _load_manifest_jsonl(path)
    return _dedupe_manifest_records(records)


def load_raw_run_manifests(raw_root: str | Path) -> list[dict[str, Any]]:
    root = Path(raw_root)
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    manifest_paths = sorted(root.glob("**/cross_device/manifests/runs.jsonl"))
    for manifest_path in manifest_paths:
        protocol_root = manifest_path.parents[1]
        for record in _load_manifest_jsonl(manifest_path):
            records.append(_remap_manifest_paths(record, protocol_root))
    return _dedupe_manifest_records(records)


def matching_manifests(
    manifests: list[dict[str, Any]],
    *,
    stage: str,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
    target_profile_instance_id: str | None = None,
    replay_device_id: str | None = None,
    replay_host_instance_id: str | None = None,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for record in manifests:
        if record["stage"] != stage:
            continue
        if not manifest_contributes_to(record, experiment_id):
            continue
        if record["policy"] != row["policy"] or record["env"] != row["env"]:
            continue
        if record["target_device_id"] != target_device_id:
            continue
        if target_profile_instance_id is not None and record["target_profile_instance_id"] != target_profile_instance_id:
            continue
        if replay_device_id is not None and record["replay_device_id"] != replay_device_id:
            continue
        if replay_host_instance_id is not None and record["instance_id"] != replay_host_instance_id:
            continue
        if record["status"] != "completed":
            continue
        matches.append(record)
    return sorted(matches, key=_record_sort_timestamp)


def skipped_notes_if_missing(
    manifests: list[dict[str, Any]],
    *,
    matches: list[dict[str, Any]],
    stage: str,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
    target_profile_instance_id: str | None = None,
    replay_device_id: str | None = None,
    replay_host_instance_id: str | None = None,
) -> str:
    if matches:
        return ""
    return matching_skipped_notes(
        manifests,
        stage=stage,
        experiment_id=experiment_id,
        row=row,
        target_device_id=target_device_id,
        target_profile_instance_id=target_profile_instance_id,
        replay_device_id=replay_device_id,
        replay_host_instance_id=replay_host_instance_id,
    )


def matching_skipped_notes(
    manifests: list[dict[str, Any]],
    *,
    stage: str,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
    target_profile_instance_id: str | None = None,
    replay_device_id: str | None = None,
    replay_host_instance_id: str | None = None,
) -> str:
    notes: list[str] = []
    for record in manifests:
        if record["stage"] != stage:
            continue
        if not manifest_contributes_to(record, experiment_id):
            continue
        if record["policy"] != row["policy"] or record["env"] != row["env"]:
            continue
        if record["target_device_id"] != target_device_id:
            continue
        if target_profile_instance_id is not None and record["target_profile_instance_id"] != target_profile_instance_id:
            continue
        if replay_device_id is not None and record["replay_device_id"] != replay_device_id:
            continue
        if replay_host_instance_id is not None and record["instance_id"] != replay_host_instance_id:
            continue
        if record["status"] not in {"skipped", "failed"}:
            continue
        note = record["skip_reason"] or record["notes"]
        if note:
            notes.append(str(note))
    return "; ".join(dict.fromkeys(notes))


def profile_instances_for_device(
    manifests: list[dict[str, Any]],
    *,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
    stages: tuple[str, ...],
) -> list[str]:
    instances = {
        str(record["target_profile_instance_id"])
        for record in manifests
        if record["stage"] in stages
        and manifest_contributes_to(record, experiment_id)
        and record["policy"] == row["policy"]
        and record["env"] == row["env"]
        and record["target_device_id"] == target_device_id
    }
    return sorted(instances)


def profile_instances_for_cross_pair(
    manifests: list[dict[str, Any]],
    *,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
    replay_device_id: str,
) -> list[str]:
    instances = {
        str(record["target_profile_instance_id"])
        for record in manifests
        if record["stage"] in {"measure", "sim_cross"}
        and manifest_contributes_to(record, experiment_id)
        and record["policy"] == row["policy"]
        and record["env"] == row["env"]
        and record["target_device_id"] == target_device_id
        and (record["stage"] == "measure" or record["replay_device_id"] == replay_device_id)
    }
    return sorted(instances)


def replay_host_instances_for_cross_pair(
    manifests: list[dict[str, Any]],
    *,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
    target_profile_instance_id: str | None,
    replay_device_id: str,
) -> list[str]:
    instances = {
        str(record["instance_id"])
        for record in manifests
        if record["stage"] == "sim_cross"
        and manifest_contributes_to(record, experiment_id)
        and record["policy"] == row["policy"]
        and record["env"] == row["env"]
        and record["target_device_id"] == target_device_id
        and record["target_profile_instance_id"] == target_profile_instance_id
        and record["replay_device_id"] == replay_device_id
    }
    return sorted(instances)


def manifest_contributes_to(record: dict[str, Any], experiment_id: str) -> bool:
    experiment_ids = record["experiment_ids"]
    if isinstance(experiment_ids, list):
        return experiment_id in {str(item) for item in experiment_ids}
    return record["experiment_id"] == experiment_id


def _load_manifest_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Manifest record must be a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def _remap_manifest_paths(record: dict[str, Any], protocol_root: Path) -> dict[str, Any]:
    remapped = dict(record)
    for key in ("output_dir", "profile_path", "generated_config_path"):
        value = remapped[key]
        if not value:
            continue
        remapped[key] = _remap_protocol_path(str(value), protocol_root)
    return remapped


def _remap_protocol_path(value: str, protocol_root: Path) -> str:
    path = Path(value)
    suffix = _protocol_path_suffix(path.parts)
    if not suffix:
        return value
    return str(protocol_root.joinpath(*suffix))


def _protocol_path_suffix(parts: tuple[str, ...]) -> tuple[str, ...]:
    for marker in ("cross_device",):
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == marker:
                return tuple(parts[index + 1 :])
    for marker in ("runs", "profiles", "generated_configs", "manifests"):
        if marker in parts:
            index = parts.index(marker)
            return tuple(parts[index:])
    return ()


def _dedupe_manifest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed_by_run_key: dict[tuple[str, str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    non_completed_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        run_key = (
            str(record["stage"]),
            str(record["experiment_id"]),
            str(record["policy"]),
            str(record["env"]),
            str(record["target_device_id"]),
            str(record["target_profile_instance_id"]),
            str(record["replay_device_id"]),
            str(record["instance_id"]),
            str(record["seed"]),
        )
        if record["status"] == "completed":
            existing = completed_by_run_key.get(run_key)
            if existing is None or _record_sort_timestamp(record) >= _record_sort_timestamp(existing):
                completed_by_run_key[run_key] = record
            continue
        skipped_key = (
            *run_key,
            str(record["status"]),
            str(record["skip_reason"] or record["notes"]),
            str(record["runtime_env_detected_raw"] or ""),
        )
        non_completed_by_key[skipped_key] = record
    return [*non_completed_by_key.values(), *completed_by_run_key.values()]


def _record_sort_timestamp(record: dict[str, Any]) -> str:
    return str(record["completed_at"] or record["started_at"])


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
