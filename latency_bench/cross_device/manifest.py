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
from latency_bench.cross_device.instances import gpu_uuid_hash_from_instance_id
from latency_bench.cross_device.planning import ProtocolTask
from latency_bench.latency.hardware_info import collect_hardware_fingerprint
from latency_bench.utils.io import load_jsonl_records, write_json


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
    eval_cmd: list[str],
    profile_cmd: list[str] | None,
) -> dict[str, Any]:
    replay_device = task.replay_device
    gpu_uuid_hash = None
    if task.instance_origin == "gpu_uuid_hash":
        gpu_uuid_hash = gpu_uuid_hash_from_instance_id(task.instance_id)
    return {
        "protocol_id": spec["protocol_id"],
        "stage": task.stage,
        "experiment_ids": task.experiment_ids,
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
        "latency_seed_set": task.latency_seed_set,
        "latency_seed": task.latency_seed,
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
    skip_reason: str,
    out: Path,
) -> None:
    manifest = build_base_manifest(task, spec, detected, _eval_command(task), None)
    manifest["status"] = "skipped"
    manifest["completed_at"] = now_iso()
    manifest["skip_reason"] = skip_reason
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


def load_run_manifests(root: str | Path, *, raw_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Load deduped run manifests from a protocol root or a downloaded raw tree.

    With `raw_root`, every `**/cross_device/manifests/runs.jsonl` under it is
    read and record paths are remapped into that raw tree; otherwise the single
    `manifests/runs.jsonl` under `root` is read.
    """
    records: list[dict[str, Any]] = []
    if raw_root is not None:
        for manifest_path in sorted(Path(raw_root).glob("**/cross_device/manifests/runs.jsonl")):
            protocol_root = manifest_path.parents[1]
            for record in load_jsonl_records(manifest_path):
                records.append(_remap_manifest_paths(record, protocol_root))
    else:
        path = Path(root) / "manifests" / "runs.jsonl"
        if path.exists():
            records = load_jsonl_records(path)
    return _dedupe_manifest_records(records)


def _matches(
    record: dict[str, Any],
    *,
    stage: str,
    experiment_id: str,
    row: dict[str, Any],
    target_device_id: str,
    target_profile_instance_id: str | None = None,
    replay_device_id: str | None = None,
    replay_host_instance_id: str | None = None,
) -> bool:
    if record["stage"] != stage:
        return False
    if not manifest_contributes_to(record, experiment_id):
        return False
    if record["policy"] != row["policy"] or record["env"] != row["env"]:
        return False
    if record["target_device_id"] != target_device_id:
        return False
    if target_profile_instance_id is not None and record["target_profile_instance_id"] != target_profile_instance_id:
        return False
    if replay_device_id is not None and record["replay_device_id"] != replay_device_id:
        return False
    if replay_host_instance_id is not None and record["instance_id"] != replay_host_instance_id:
        return False
    return True


def matching_manifests(manifests: list[dict[str, Any]], **query: Any) -> list[dict[str, Any]]:
    """Return completed run manifests matching the query, oldest first."""
    matches = [
        record for record in manifests if record["status"] == "completed" and _matches(record, **query)
    ]
    return sorted(matches, key=_record_sort_timestamp)


def matching_skipped_notes(manifests: list[dict[str, Any]], **query: Any) -> str:
    """Join the skip/error reasons of skipped or failed runs matching the query."""
    notes: list[str] = []
    for record in manifests:
        if record["status"] not in {"skipped", "failed"} or not _matches(record, **query):
            continue
        note = record["skip_reason"] if record["status"] == "skipped" else record["error"]
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
    return experiment_id in {str(item) for item in record["experiment_ids"]}


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
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "cross_device":
            return tuple(parts[index + 1 :])
    return ()


def _dedupe_manifest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed_by_run_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    non_completed_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        run_key = (
            str(record["stage"]),
            tuple(str(item) for item in record["experiment_ids"]),
            str(record["policy"]),
            str(record["env"]),
            str(record["target_device_id"]),
            str(record["target_profile_instance_id"]),
            str(record["replay_device_id"]),
            str(record["instance_id"]),
            str(record["seed"]),
            str(_latency_session_seed(record)),
        )
        if record["status"] == "completed":
            existing = completed_by_run_key.get(run_key)
            if existing is None or _record_sort_timestamp(record) >= _record_sort_timestamp(existing):
                completed_by_run_key[run_key] = record
            continue
        skipped_key = (
            *run_key,
            str(record["status"]),
            str(record["skip_reason"] if record["status"] == "skipped" else record["error"]),
        )
        non_completed_by_key[skipped_key] = record
    return [*non_completed_by_key.values(), *completed_by_run_key.values()]


def _latency_session_seed(record: dict[str, Any]) -> Any:
    """Return the latency seed; legacy sim manifests used `seed` for both RNGs."""
    if "latency_seed" in record:
        return record["latency_seed"]
    if record["stage"] in {"sim_same", "sim_cross"}:
        return record["seed"]
    return None


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
