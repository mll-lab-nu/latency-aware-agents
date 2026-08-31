from __future__ import annotations

import hashlib
import json
from typing import Any

def instance_id_from_gpu_deployment(deployment: dict[str, Any]) -> str:
    topology_links = [
        {
            "gpu_a": min(link["gpu_a"], link["gpu_b"]),
            "gpu_b": max(link["gpu_a"], link["gpu_b"]),
            "link": link["link"],
        }
        for link in deployment["topology_links"]
    ]
    identity = {
        "gpus": sorted(gpu["uuid"] for gpu in deployment["gpus"]),
        "topology_links": sorted(
            topology_links, key=lambda link: (link["gpu_a"], link["gpu_b"])
        ),
    }
    encoded = json.dumps(identity, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"instance_{digest[:16]}"
