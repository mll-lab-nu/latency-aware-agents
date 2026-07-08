from __future__ import annotations

import inspect
from importlib import metadata
from typing import Any

import numpy as np


SUPPORTED_SAMPLE_FACTORY_VERSION = "2.1.1"


def validate_sample_factory_version(sample_factory_module: Any, *, compat_label: str) -> None:
    try:
        installed_version = metadata.version("sample-factory")
    except metadata.PackageNotFoundError:
        installed_version = str(getattr(sample_factory_module, "__version__", "") or "")

    if installed_version != SUPPORTED_SAMPLE_FACTORY_VERSION:
        raise RuntimeError(
            f"Unsupported Sample Factory version for {compat_label}: "
            f"expected {SUPPORTED_SAMPLE_FACTORY_VERSION}, got {installed_version or 'unknown'}"
        )


def validate_signature(callable_obj: Any, *, expected_params: tuple[str, ...], label: str) -> None:
    signature = inspect.signature(callable_obj)
    actual_params = tuple(signature.parameters)
    if actual_params != expected_params:
        raise RuntimeError(
            f"Sample Factory {label} signature changed: expected {expected_params}, got {actual_params}"
        )


def is_enabled_for_sample_factory_env(obj: Any, enabled_env_names: set[str]) -> bool:
    cfg = getattr(obj, "cfg", None)
    env_name = cfg_value(cfg, "env")
    if env_name is None:
        env_name = cfg_value(cfg, "env_name")
    if env_name is None:
        env_name = cfg_value(cfg, "full_env_name")
    return env_name in enabled_env_names


def cfg_value(cfg: Any, key: str) -> str | None:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        value = cfg.get(key)
    else:
        value = getattr(cfg, key, None)
    if value is None:
        return None
    return str(value)


def done_indices(done_mask: Any) -> list[int]:
    done_array = as_numpy_array(done_mask).astype(bool, copy=False).reshape(-1)
    return [int(idx) for idx, done in enumerate(done_array.tolist()) if done]


def as_numpy_array(value: Any) -> np.ndarray:
    item = value
    if hasattr(item, "detach") and callable(item.detach):
        item = item.detach()
    if hasattr(item, "cpu") and callable(item.cpu):
        item = item.cpu()
    if hasattr(item, "numpy") and callable(item.numpy):
        item = item.numpy()
    return np.asarray(item)
