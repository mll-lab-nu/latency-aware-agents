from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


LEGACY_INSTANCE_ID = "instance_legacy_01"


@dataclass(frozen=True)
class MigrationAction:
    source: Path
    destination: Path


def discover_migration_actions(root: Path) -> list[MigrationAction]:
    raw_root = root / "raw"
    actions: list[MigrationAction] = []
    for device_dir in sorted(raw_root.iterdir()):
        if not device_dir.is_dir():
            continue
        for child in sorted(device_dir.iterdir()):
            if child.name == "instances" or not child.is_dir():
                continue
            actions.append(
                MigrationAction(
                    source=child,
                    destination=device_dir / "instances" / LEGACY_INSTANCE_ID / child.name,
                )
            )
    return actions


def apply_migration(root: Path, actions: list[MigrationAction]) -> None:
    manifest_records = []
    for action in actions:
        action.destination.parent.mkdir(parents=True, exist_ok=True)
        if action.destination.exists():
            raise FileExistsError(action.destination)
        shutil.move(str(action.source), str(action.destination))
        manifest_records.append(
            {
                "source": action.source.as_posix(),
                "destination": action.destination.as_posix(),
                "instance_id": LEGACY_INSTANCE_ID,
                "instance_origin": "legacy_unlabeled",
                "gpu_uuid_hash": None,
            }
        )
    manifest_path = root / "raw_instance_migration_manifest.json"
    manifest_path.write_text(json.dumps(manifest_records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Move legacy raw/<device>/<runtime> trees under instances/instance_legacy_01."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    actions = discover_migration_actions(args.root)
    for action in actions:
        print(f"{action.source} -> {action.destination}")
    if args.apply:
        apply_migration(args.root, actions)


if __name__ == "__main__":
    main()
