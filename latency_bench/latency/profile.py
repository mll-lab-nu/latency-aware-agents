from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


SUPPORTED_PROFILE_TYPES = {"micro_profile", "rollout_profile"}

REQUIRED_PROFILE_FIELDS = (
    "schema_version",
    "profile_id",
    "profile_type",
    "policy_id",
    "env_id",
    "target_device",
    "fps",
    "frame_ms",
    "n_warmup",
    "n_samples",
    "batch_size",
    "trace_path",
    "summary",
)


def save_profile_manifest(profile: Mapping[str, Any], path: str | Path) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(dict(profile), f, indent=2, sort_keys=True)
        f.write("\n")


def load_profile_manifest(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)
    if not isinstance(profile, dict):
        raise ValueError(f"Profile manifest must be a JSON object: {profile_path}")
    _validate_profile_manifest(profile, profile_path)
    return profile


def write_latency_samples(samples: Iterable[float], path: str | Path) -> None:
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f:
        for index, sample in enumerate(samples):
            latency_ms = _coerce_latency_ms(sample, f"{trace_path}:sample[{index}]")
            f.write(json.dumps({"latency_ms": latency_ms}, sort_keys=True) + "\n")


def load_latency_samples(path: str | Path) -> list[float]:
    trace_path = Path(path)
    samples: list[float] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise ValueError(f"Latency sample record must be a JSON object at {trace_path}:{line_number}")
            if "latency_ms" not in record:
                raise ValueError(
                    f"Missing latency_ms at {trace_path}:{line_number}"
                )
            samples.append(_coerce_latency_ms(record["latency_ms"], f"{trace_path}:{line_number}"))
    if not samples:
        raise ValueError(f"Latency sample trace is empty: {trace_path}")
    return samples


def _validate_profile_manifest(profile: dict[str, Any], profile_path: Path) -> None:
    for field in REQUIRED_PROFILE_FIELDS:
        if field not in profile:
            raise ValueError(f"Missing required profile field '{field}' in {profile_path}")
    if profile["schema_version"] != 1:
        raise ValueError(f"Unsupported profile schema_version {profile['schema_version']!r} in {profile_path}")
    if profile["profile_type"] not in SUPPORTED_PROFILE_TYPES:
        raise ValueError(
            f"Unsupported profile_type {profile['profile_type']!r} in {profile_path}; "
            f"expected one of {sorted(SUPPORTED_PROFILE_TYPES)}"
        )
    fps = float(profile["fps"])
    frame_ms = float(profile["frame_ms"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite in {profile_path}")
    if not math.isfinite(frame_ms) or frame_ms <= 0:
        raise ValueError(f"frame_ms must be positive and finite in {profile_path}")
    expected_frame_ms = 1000.0 / fps
    if not math.isclose(frame_ms, expected_frame_ms, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"frame_ms {frame_ms!r} does not match fps-derived frame_ms {expected_frame_ms!r} in {profile_path}"
        )


def _coerce_latency_ms(value: Any, location: str) -> float:
    latency_ms = float(value)
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError(f"latency_ms must be finite and non-negative at {location}")
    return latency_ms
