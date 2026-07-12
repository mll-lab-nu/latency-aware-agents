from __future__ import annotations

import hashlib
import os
from typing import Any


INSTANCE_ID_ENV = "CROSS_DEVICE_INSTANCE_ID"


def resolve_instance_id(fingerprint: dict[str, Any]) -> str:
    override = os.environ.get(INSTANCE_ID_ENV)
    if override:
        return override
    return instance_id_from_gpu_uuid(fingerprint["gpu"]["uuid"])


def instance_id_from_gpu_uuid(gpu_uuid: str) -> str:
    digest = hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()
    return f"instance_{digest[:8]}"
