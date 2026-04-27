from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.core.types import LatencySourceType, PersistenceMode, StaleActionMode


REQUIRED_TOP_LEVEL_KEYS = (
    "experiment",
    "env",
    "control",
    "latency",
    "scheduler",
    "policy",
    "logging",
)


def merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "__base__":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict:
    config = _load_config_unvalidated(Path(path))
    validate_config(config)
    return config


def _load_config_unvalidated(path: Path) -> dict:
    import yaml

    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        current = yaml.safe_load(f) or {}
    if not isinstance(current, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    base_ref = current.get("__base__")
    if not base_ref:
        return {k: v for k, v in current.items() if k != "__base__"}

    if isinstance(base_ref, (str, Path)):
        base_paths = [base_ref]
    elif isinstance(base_ref, list):
        base_paths = base_ref
    else:
        raise ValueError("__base__ must be a path string or list of path strings")

    merged: dict[str, Any] = {}
    for base_item in base_paths:
        base_path = Path(base_item)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = merge_dicts(merged, _load_config_unvalidated(base_path))

    return merge_dicts(merged, current)


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    experiment = _require_dict(config, "experiment")
    env = _require_dict(config, "env")
    control = _require_dict(config, "control")
    latency = _require_dict(config, "latency")
    scheduler = _require_dict(config, "scheduler")
    policy = _require_dict(config, "policy")
    logging = _require_dict(config, "logging")

    _require_field(experiment, "name")
    _require_field(env, "name")
    fps_env = _require_field(env, "fps_env")
    if float(fps_env) <= 0:
        raise ValueError("env.fps_env must be positive")
    _require_field(env, "noop_action")

    if "frequency_hz" in control and float(control["frequency_hz"]) <= 0:
        raise ValueError("control.frequency_hz must be positive")
    if "observe_every_n_frames" in control and int(control["observe_every_n_frames"]) <= 0:
        raise ValueError("control.observe_every_n_frames must be positive")

    latency_type = _require_field(latency, "type")
    _validate_enum_value(latency_type, LatencySourceType, "latency.type")
    if latency_type == LatencySourceType.FIXED.value:
        _require_field(latency, "fixed_latency_ms")
    if latency_type == LatencySourceType.TRACE_REPLAY.value:
        _require_field(latency, "trace_path")
    if latency_type == LatencySourceType.EMPIRICAL_DISTRIBUTION.value:
        values = _require_field(latency, "latency_values_ms")
        if not values:
            raise ValueError("latency.latency_values_ms cannot be empty")

    if "persistence_mode" in scheduler:
        _validate_enum_value(scheduler["persistence_mode"], PersistenceMode, "scheduler.persistence_mode")
    if "stale_action_mode" in scheduler:
        _validate_enum_value(scheduler["stale_action_mode"], StaleActionMode, "scheduler.stale_action_mode")

    _require_field(policy, "type")
    _require_field(logging, "output_dir")


def _require_dict(config: dict, key: str) -> dict:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_field(config: dict, key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required field: {key}")
    return config[key]


def _validate_enum_value(value: Any, enum_type: type, field_name: str) -> None:
    allowed = {item.value for item in enum_type}
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")

