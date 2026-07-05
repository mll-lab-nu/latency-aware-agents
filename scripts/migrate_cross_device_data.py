"""THROWAWAY one-time migration of cross-device data to the current formats.

Delete this script after the existing local + HF data has been converted and
re-uploaded. It converts data written by the pre-instance-id protocol code:

1. profiles: drop `e2e_summary`/`e2e_trace_path` (asserted equal to
   `summary`/`trace_path`) and `diagnostics_path`; relocate
   `profiles*/<policy>/<env>/<profile_id>/` to
   `profiles*/<model_type>/<env_name>/<device_id>/<instance_id>/`.
2. run dirs: rename legacy record filenames to `episode_metrics.jsonl`/
   `latencies.jsonl`/`actions.jsonl`; nest legacy top-level episode metadata
   keys under `metadata`.
3. manifests (`manifests/runs.jsonl` + per-run `manifest.json`): ensure
   `experiment_ids` is a list; drop `experiment_id`/`panel`/
   `runtime_env_matched_alias`; fill `instance_id`/`instance_origin`/
   `gpu_uuid_hash`/`target_profile_instance_id`; copy skipped `notes` into
   `skip_reason`; rewrite `profile_path` to the new profile layout.
4. raw tree: move `raw/<device>/<runtime>/cross_device/**` to
   `raw/<device>/instances/<instance_id>/<runtime>/cross_device/**`.

No record in the existing data carries a GPU UUID, so every device needs an
explicit `--instance-map <device_id>=<instance_id>` entry.

Usage (verify on a copy first):
    python scripts/migrate_cross_device_data.py \
        --raw-root /tmp/migrated/_hf_raw/raw \
        --root /tmp/migrated \
        --instance-map rtx3090=instance_3090aaaa --instance-map a100=instance_a100aaaa ...
Then re-upload with `python -m latency_bench.cross_device.sync upload`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from latency_bench.cross_device.instances import instance_id_from_gpu_uuid


LEGACY_RUN_FILENAMES = {
    "metrics.jsonl": "episode_metrics.jsonl",
    "episodes.jsonl": "episode_metrics.jsonl",
    "latency_records.jsonl": "latencies.jsonl",
    "action_records.jsonl": "actions.jsonl",
}

LEGACY_EPISODE_METADATA_KEYS = (
    "mode",
    "policy_id",
    "env_id",
    "profile_id",
    "target_device",
    "replay_host",
    "env_fps",
    "frame_ms",
    "latency_type",
)

PROFILE_ROOT_NAMES = ("profiles", "profiles_smoke")


def log(action: str, detail: str, changes: list[str]) -> None:
    changes.append(f"{action}: {detail}")
    print(f"{action}: {detail}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def instance_for_device(device_id: str, instance_map: dict[str, str], *, context: str) -> str:
    if device_id not in instance_map:
        raise SystemExit(
            f"no instance id for device {device_id!r} ({context}); "
            f"pass --instance-map {device_id}=instance_xxxxxxxx"
        )
    return instance_map[device_id]


def migrate_profile_manifest(path: Path, changes: list[str], *, require_worker_count: bool) -> dict[str, Any]:
    manifest = read_json(path)
    dirty = False
    if "e2e_summary" in manifest:
        assert manifest["e2e_summary"] == manifest["summary"], f"e2e_summary != summary in {path}"
        del manifest["e2e_summary"]
        dirty = True
    if "e2e_trace_path" in manifest:
        assert manifest["e2e_trace_path"] == manifest["trace_path"], f"e2e_trace_path != trace_path in {path}"
        del manifest["e2e_trace_path"]
        dirty = True
    if "diagnostics_path" in manifest:
        del manifest["diagnostics_path"]
        dirty = True
    if require_worker_count and manifest.get("inference_worker_count") is None:
        raise SystemExit(f"profile missing inference_worker_count (re-measure required): {path}")
    if dirty:
        write_json(path, manifest)
        log("profile-keys", str(path), changes)
    return manifest


def relocate_profiles(protocol_root: Path, instance_map: dict[str, str], changes: list[str]) -> None:
    for root_name in PROFILE_ROOT_NAMES:
        profiles_root = protocol_root / root_name
        if not profiles_root.exists():
            continue
        # profiles_smoke holds debug artifacts; don't hard-require worker counts there.
        require_worker_count = root_name == "profiles"
        # New layout is 4 dirs deep (model/env/device/instance); old is 3.
        for profile_json in sorted(profiles_root.glob("*/*/*/*/profile.json")):
            migrate_profile_manifest(profile_json, changes, require_worker_count=require_worker_count)
        for profile_json in sorted(profiles_root.glob("*/*/*/profile.json")):
            manifest = migrate_profile_manifest(profile_json, changes, require_worker_count=require_worker_count)
            device_id = str(manifest["target_device"])
            hardware = profile_json.parent / "hardware.json"
            gpu_uuid = ""
            if hardware.exists():
                gpu_uuid = str(((read_json(hardware).get("gpu")) or {}).get("uuid") or "").strip()
            instance_id = (
                instance_id_from_gpu_uuid(gpu_uuid)
                if gpu_uuid
                else instance_for_device(device_id, instance_map, context=str(profile_json))
            )
            new_dir = (
                profiles_root
                / str(manifest["model_type"])
                / str(manifest["env_id"])
                / device_id
                / instance_id
            )
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            if new_dir.exists():
                raise SystemExit(f"refusing to overwrite existing profile dir: {new_dir}")
            old_dir = profile_json.parent
            old_dir.rename(new_dir)
            log("profile-move", f"{old_dir} -> {new_dir}", changes)
            # Remove now-empty old policy/env dirs.
            for parent in (old_dir.parent, old_dir.parent.parent):
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()


def migrate_run_dirs(protocol_root: Path, changes: list[str]) -> None:
    runs_root = protocol_root / "runs"
    if not runs_root.exists():
        return
    for legacy_name, canonical_name in LEGACY_RUN_FILENAMES.items():
        for legacy in sorted(runs_root.rglob(legacy_name)):
            canonical = legacy.parent / canonical_name
            if canonical.exists():
                raise SystemExit(f"both {legacy} and {canonical} exist; resolve manually")
            legacy.rename(canonical)
            log("run-file-rename", f"{legacy} -> {canonical}", changes)
    for metrics_path in sorted(runs_root.rglob("episode_metrics.jsonl")):
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        dirty = False
        for record in records:
            if isinstance(record.get("metadata"), dict):
                continue
            metadata = {key: record.pop(key) for key in LEGACY_EPISODE_METADATA_KEYS if key in record}
            record["metadata"] = metadata
            dirty = True
        if dirty:
            metrics_path.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
            )
            log("episode-metadata-nest", str(metrics_path), changes)


def rewrite_profile_path(value: str, record: dict[str, Any], instance_map: dict[str, str]) -> str:
    if not value:
        return value
    parts = Path(value).parts
    root_index = next((i for i, part in enumerate(parts) if part in PROFILE_ROOT_NAMES), None)
    if root_index is None:
        return value
    suffix = parts[root_index + 1 :]
    if len(suffix) == 5:  # already model/env/device/instance/profile.json
        return value
    device_id = str(record["target_device_id"])
    instance_id = instance_for_device(device_id, instance_map, context=f"profile_path {value!r}")
    return str(
        Path(*parts[: root_index + 1])
        / str(record["model_type"])
        / str(record["env_name"])
        / device_id
        / instance_id
        / "profile.json"
    )


def migrate_manifest_record(record: dict[str, Any], instance_map: dict[str, str]) -> dict[str, Any]:
    record = dict(record)
    if "experiment_ids" not in record:
        record["experiment_ids"] = [record["experiment_id"]]
    for key in ("experiment_id", "panel", "runtime_env_matched_alias"):
        record.pop(key, None)

    device_id = str(record["target_device_id"])
    gpu_uuid = str(((record.get("hardware_fingerprint") or {}).get("gpu") or {}).get("uuid") or "").strip()
    context = f"manifest {record['stage']} {record.get('output_dir')}"
    if record.get("instance_id") is None:
        if gpu_uuid:
            record["instance_id"] = instance_id_from_gpu_uuid(gpu_uuid)
            record["instance_origin"] = "gpu_uuid_hash"
            record["gpu_uuid_hash"] = record["instance_id"].removeprefix("instance_")
        else:
            host_device = device_id if record["stage"] != "sim_cross" else str(record["replay_device_id"])
            record["instance_id"] = instance_for_device(host_device, instance_map, context=context)
            record["instance_origin"] = "manual_override"
            record["gpu_uuid_hash"] = None
    record.setdefault("instance_origin", "manual_override")
    record.setdefault("gpu_uuid_hash", None)
    if record.get("target_profile_instance_id") is None:
        record["target_profile_instance_id"] = instance_for_device(device_id, instance_map, context=context)

    if record["status"] == "skipped" and not record.get("skip_reason"):
        record["skip_reason"] = str(record.get("notes") or "")
    if record.get("notes") and record["notes"] == record.get("skip_reason"):
        record["notes"] = ""

    record["profile_path"] = rewrite_profile_path(str(record.get("profile_path") or ""), record, instance_map)
    return record


def migrate_manifests(protocol_root: Path, instance_map: dict[str, str], changes: list[str]) -> None:
    runs_jsonl = protocol_root / "manifests" / "runs.jsonl"
    if runs_jsonl.exists():
        records = [json.loads(line) for line in runs_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        migrated = []
        for record in records:
            if "model_type" not in record:
                # Pre-model_type debug record; unusable under the new schema.
                log("manifest-DROP", f"{runs_jsonl}: {record['status']} {record.get('output_dir')}", changes)
                continue
            migrated.append(migrate_manifest_record(record, instance_map))
        if migrated != records:
            runs_jsonl.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in migrated), encoding="utf-8"
            )
            log("manifest-jsonl", f"{runs_jsonl} ({len(migrated)} records)", changes)
    runs_root = protocol_root / "runs"
    if not runs_root.exists():
        return
    for manifest_path in sorted(runs_root.rglob("manifest.json")):
        record = read_json(manifest_path)
        if "model_type" not in record:
            log("manifest-SKIP", f"pre-model_type per-run manifest left as-is: {manifest_path}", changes)
            continue
        migrated = migrate_manifest_record(record, instance_map)
        if migrated != record:
            write_json(manifest_path, migrated)
            log("manifest-json", str(manifest_path), changes)


def migrate_protocol_root(protocol_root: Path, instance_map: dict[str, str], changes: list[str]) -> None:
    print(f"== migrating protocol root: {protocol_root}")
    relocate_profiles(protocol_root, instance_map, changes)
    migrate_run_dirs(protocol_root, changes)
    migrate_manifests(protocol_root, instance_map, changes)


def migrate_raw_tree(raw_root: Path, instance_map: dict[str, str], changes: list[str]) -> None:
    for device_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        device_id = device_dir.name
        for runtime_dir in sorted(path for path in device_dir.iterdir() if path.is_dir()):
            if runtime_dir.name == "instances":
                for instance_dir in sorted(path for path in runtime_dir.iterdir() if path.is_dir()):
                    for nested_runtime in sorted(path for path in instance_dir.iterdir() if path.is_dir()):
                        protocol_root = nested_runtime / "cross_device"
                        if protocol_root.exists():
                            migrate_protocol_root(protocol_root, instance_map, changes)
                continue
            protocol_root = runtime_dir / "cross_device"
            if not protocol_root.exists():
                continue
            instance_id = instance_for_device(device_id, instance_map, context=str(runtime_dir))
            new_runtime_dir = device_dir / "instances" / instance_id / runtime_dir.name
            if new_runtime_dir.exists():
                raise SystemExit(f"refusing to overwrite existing raw dir: {new_runtime_dir}")
            new_runtime_dir.parent.mkdir(parents=True, exist_ok=True)
            runtime_dir.rename(new_runtime_dir)
            log("raw-move", f"{runtime_dir} -> {new_runtime_dir}", changes)
            migrate_protocol_root(new_runtime_dir / "cross_device", instance_map, changes)


def parse_instance_map(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        for entry in item.split(","):
            entry = entry.strip()
            if not entry:
                continue
            device_id, _, instance_id = entry.partition("=")
            if not device_id or not instance_id:
                raise SystemExit(f"--instance-map entries must be <device_id>=<instance_id>: {entry!r}")
            mapping[device_id] = instance_id
    return mapping


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="Local protocol output root (e.g. results/cross_device).")
    parser.add_argument("--raw-root", type=Path, help="Downloaded raw tree root (the directory holding <device>/...).")
    parser.add_argument(
        "--instance-map",
        action="append",
        default=[],
        help="<device_id>=<instance_id> (repeatable / comma-separated) for data without a GPU UUID.",
    )
    args = parser.parse_args(argv)
    if args.root is None and args.raw_root is None:
        raise SystemExit("pass --root and/or --raw-root")
    instance_map = parse_instance_map(args.instance_map)

    changes: list[str] = []
    if args.root is not None:
        migrate_protocol_root(args.root, instance_map, changes)
    if args.raw_root is not None:
        migrate_raw_tree(args.raw_root, instance_map, changes)
    print(f"\n{len(changes)} changes applied." if changes else "\nNothing to migrate (already current).")


if __name__ == "__main__":
    main()
