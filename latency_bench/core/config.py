from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
import os
from typing import Any
import yaml


def load_factory(reference: str) -> Callable[..., Any]:
    """Load a user factory named by a ``module:function`` configuration value."""
    module_name, separator, function_name = reference.partition(":")
    if not separator or not module_name or not function_name or ":" in function_name:
        raise ValueError(f"Factory must use 'module:function', got {reference!r}")
    factory = getattr(import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"Factory {reference!r} is not callable")
    return factory


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
    path = Path(path).expanduser().resolve()
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
        merged = merge_dicts(merged, load_config(base_path))

    return merge_dicts(merged, current)


def resolve_runtime_config(config: dict[str, Any]) -> None:
    """Resolve common run defaults and filesystem inputs before execution.

    Mode-specific fields are read only by the corresponding execution path.
    Artifact roots retain HF snapshot symlinks for checkpoint sibling metadata.
    """
    defaults = {
        "evaluation": {"eval_raw_reward": False, "suites": []},
        "executor": {
            "parallel_envs": 1,
            "inference_pool": False,
            "simulated_worker_capacity": None,
            "sync_cuda": True,
            "env_backend": "auto",
        },
        "policy": {"deterministic": True},
        "logging": {
            "save_step_records": False,
            "save_action_records": False,
            "save_latency_records": False,
        },
    }
    for section, fields in defaults.items():
        for key, value in fields.items():
            if key not in config[section]:
                config[section][key] = deepcopy(value)
    evaluation = config["evaluation"]
    for key in ("episodes", "max_env_steps"):
        if key in evaluation and evaluation[key] is not None:
            evaluation[key] = int(evaluation[key])
    executor = config["executor"]
    executor["parallel_envs"] = int(executor["parallel_envs"])
    if (
        config["policy"]["type"] == "sample_factory"
        and executor["mode"] != "realtime"
        and not executor["inference_pool"]
        and "inference_devices" not in executor
    ):
        executor["inference_devices"] = ["auto"]
    if "video" not in config["logging"]:
        config["logging"]["video"] = {"enabled": False}
    if executor["mode"] != "realtime" and config["logging"]["video"]["enabled"]:
        config["logging"]["save_step_records"] = True
    path_fields = {
        "policy": ("backbone_path",),
        "logging": ("output_dir",),
    }
    if executor["mode"] == "simulated":
        path_fields["latency"] = ("profile_path",)
        for suite in evaluation["suites"]:
            if "profile_path" in suite["latency"]:
                suite["latency"]["profile_path"] = os.path.abspath(
                    Path(suite["latency"]["profile_path"]).expanduser()
                )
    for section, fields in path_fields.items():
        for key in fields:
            if key in config[section] and config[section][key] is not None:
                config[section][key] = os.path.abspath(
                    Path(config[section][key]).expanduser()
                )
