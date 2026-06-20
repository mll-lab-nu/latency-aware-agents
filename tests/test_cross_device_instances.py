from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_instance_id_uses_gpu_uuid_hash(monkeypatch):
    from scripts.cross_device.instances import resolve_instance_id

    monkeypatch.delenv("CROSS_DEVICE_INSTANCE_ID", raising=False)
    gpu_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    fingerprint = {"gpu": {"uuid": gpu_uuid}}

    expected = "instance_" + hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()[:8]

    assert resolve_instance_id(fingerprint) == expected


def test_instance_id_env_override_wins(monkeypatch):
    from scripts.cross_device.instances import resolve_instance_id

    monkeypatch.setenv("CROSS_DEVICE_INSTANCE_ID", "instance_manual")

    assert resolve_instance_id({"gpu": {"uuid": "GPU-unused"}}) == "instance_manual"


def test_instance_id_requires_uuid_without_override(monkeypatch):
    from scripts.cross_device.instances import resolve_instance_id

    monkeypatch.delenv("CROSS_DEVICE_INSTANCE_ID", raising=False)

    with pytest.raises(RuntimeError, match="GPU UUID"):
        resolve_instance_id({"gpu": {}})


def test_normalize_raw_instance_layout_moves_legacy_device_dirs(tmp_path: Path):
    from scripts.cross_device.normalize_raw_instance_layout import apply_migration, discover_migration_actions

    root = tmp_path / "repo"
    legacy = root / "raw" / "rtx3090" / "openvla" / "cross_device" / "runs"
    legacy.mkdir(parents=True)
    (legacy / "marker.txt").write_text("legacy\n", encoding="utf-8")

    actions = discover_migration_actions(root)

    assert [(action.source, action.destination) for action in actions] == [
        (
            root / "raw" / "rtx3090" / "openvla",
            root / "raw" / "rtx3090" / "instances" / "instance_legacy_01" / "openvla",
        )
    ]

    apply_migration(root, actions)

    assert not (root / "raw" / "rtx3090" / "openvla").exists()
    assert (
        root
        / "raw"
        / "rtx3090"
        / "instances"
        / "instance_legacy_01"
        / "openvla"
        / "cross_device"
        / "runs"
        / "marker.txt"
    ).read_text(encoding="utf-8") == "legacy\n"
    manifest = root / "raw_instance_migration_manifest.json"
    assert manifest.exists()


def test_profile_command_records_instance_metadata(tmp_path: Path):
    from scripts.cross_device.config import _profile_command

    task = SimpleNamespace(
        output_dir=tmp_path / "runs" / "measure",
        profile_output_dir=tmp_path / "profiles" / "openvla-flappy" / "flappy" / "L_3090",
        row={"policy_type": "starvla", "model_type": "openvla", "policy": "openvla-flappy", "env_name": "flappy"},
        target_device={"id": "rtx3090", "profile_id": "L_3090"},
        instance_id="instance_abcd1234",
        instance_origin="gpu_uuid_hash",
    )
    generated_config = {
        "policy": {"checkpoint_path": "/tmp/checkpoint.pt"},
        "env": {"env_fps": 30},
    }

    cmd = _profile_command(task, {}, False, generated_config)
    metadata = json.loads(cmd[cmd.index("--metadata-json") + 1])

    assert metadata["hardware_class"] == "rtx3090"
    assert metadata["instance_id"] == "instance_abcd1234"
    assert metadata["instance_origin"] == "gpu_uuid_hash"
    assert metadata["gpu_uuid_hash"] == "abcd1234"
