from __future__ import annotations

import re
from pathlib import Path


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_")


def profile_dir(
    profiles_dir: Path,
    *,
    model_type: str,
    env_name: str,
    device_id: str,
    instance_id: str,
) -> Path:
    return Path(profiles_dir) / slugify(model_type) / slugify(env_name) / str(device_id) / str(instance_id)


def profile_path(
    profiles_dir: Path,
    *,
    model_type: str,
    env_name: str,
    device_id: str,
    instance_id: str,
) -> Path:
    return profile_dir(
        profiles_dir,
        model_type=model_type,
        env_name=env_name,
        device_id=device_id,
        instance_id=instance_id,
    ) / "profile.json"
