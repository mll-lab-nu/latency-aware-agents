"""OpenVLA dependency checks used by dry-run and execution.

These checks report missing local model assets before a long realtime run starts.
They do not download checkpoints or modify configs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config
from latency_bench.policy.openvla_statistics import openvla_dataset_statistics_path
from scripts.cross_device.config import (
    OPENVLA_POLICY_TYPES,
    _base_config_path,
    _policy_type,
)


def _config_dependency_reason(row: dict[str, Any]) -> str | None:
    try:
        path = _base_config_path(row)
    except FileNotFoundError as exc:
        return str(exc)
    if not path.exists():
        return f"missing base_config: {path}"
    try:
        config = load_config(path)
    except Exception as exc:
        return f"config load failed: {exc}"
    return None


def _openvla_dependency_errors(config: dict[str, Any]) -> list[str]:
    """Return missing local OpenVLA assets for one generated config.

    Args:
        config: Generated evaluation config after env overrides.

    Returns:
        Human-readable dependency errors, or an empty list when dependencies are
        available. The function only checks local files and never downloads.
    """
    policy = config.get("policy", {})
    policy_type = _policy_type(config)
    if policy_type not in OPENVLA_POLICY_TYPES:
        return []
    errors: list[str] = []
    base_model_value = str(policy.get("base_model_path") or "").strip()
    checkpoint_value = str(policy.get("checkpoint_path") or "").strip()
    unnorm_key = str(policy.get("unnorm_key") or "").strip()
    if not base_model_value:
        errors.append("missing OpenVLA policy.base_model_path")
        base_model_path = None
    else:
        base_model_path = Path(base_model_value)
        if not base_model_path.exists():
            errors.append(f"missing OpenVLA policy.base_model_path: {base_model_path}")
    statistics_path = (
        openvla_dataset_statistics_path(base_model_path, policy_type)
        if base_model_path is not None
        else None
    )
    if statistics_path is not None:
        if not statistics_path.is_file():
            errors.append(f"missing OpenVLA dataset statistics file: {statistics_path}")
        else:
            try:
                statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"failed to read OpenVLA dataset statistics file at {statistics_path}: {exc}")
            else:
                if not isinstance(statistics, dict):
                    errors.append(f"OpenVLA dataset statistics file must contain a JSON object: {statistics_path}")
                elif unnorm_key not in statistics:
                    errors.append(
                        f"missing OpenVLA unnorm_key {unnorm_key!r} in dataset statistics file: {statistics_path}"
                    )
                else:
                    errors.extend(_openvla_action_stats_errors(config, statistics, unnorm_key, statistics_path))
    if not unnorm_key:
        errors.append("missing OpenVLA policy.unnorm_key")
    if not checkpoint_value:
        errors.append("missing OpenVLA policy.checkpoint_path")
    else:
        checkpoint_path = Path(checkpoint_value)
        if not checkpoint_path.exists():
            errors.append(f"missing OpenVLA policy.checkpoint_path: {checkpoint_path}")
        weights_path = checkpoint_path / "actor" / "model_state_dict" / "full_weights.pt"
        if not weights_path.is_file():
            errors.append(f"missing OpenVLA full_weights.pt: {weights_path}")
    return errors


def _openvla_action_stats_errors(
    config: dict[str, Any],
    statistics: dict[str, Any],
    unnorm_key: str,
    statistics_path: Path,
) -> list[str]:
    policy = config.get("policy", {})
    raw_action_dim = policy.get("action_dim")
    if isinstance(raw_action_dim, bool) or not isinstance(raw_action_dim, int) or raw_action_dim <= 0:
        return [f"missing or invalid OpenVLA policy.action_dim for action stats validation: {raw_action_dim!r}"]
    dataset_stats = statistics.get(unnorm_key)
    if not isinstance(dataset_stats, dict):
        return [
            f"OpenVLA stats for {unnorm_key!r} must be an object in dataset statistics file: {statistics_path}"
        ]
    action_stats = dataset_stats.get("action")
    if not isinstance(action_stats, dict):
        return [
            f"OpenVLA stats for {unnorm_key!r} must contain action stats object in dataset statistics file: {statistics_path}"
        ]
    errors: list[str] = []
    for key in ("q01", "q99", "min", "max", "mask"):
        value = action_stats.get(key)
        if value is None:
            errors.append(f"{unnorm_key} action stats missing {key} in dataset statistics file: {statistics_path}")
            continue
        try:
            length = len(value)
        except TypeError:
            errors.append(
                f"{unnorm_key} action stats {key} must be a sequence of length {raw_action_dim}, "
                f"got {type(value).__name__}"
            )
            continue
        if length != raw_action_dim:
            errors.append(
                f"{unnorm_key} action stats {key} length {length} != policy.action_dim {raw_action_dim}"
            )
    return errors


def _openvla_dependency_report(config: dict[str, Any]) -> dict[str, str]:
    """Summarize OpenVLA dependency status for tabular doctor output.

    Args:
        config: Generated or dependency-check config.

    Returns:
        String fields for base model, stats JSON, unnorm key, checkpoint,
        weights, and aggregate status.
    """
    policy = config.get("policy", {})
    policy_type = _policy_type(config)
    base_model_value = str(policy.get("base_model_path") or "").strip()
    checkpoint_value = str(policy.get("checkpoint_path") or "").strip()
    unnorm_key = str(policy.get("unnorm_key") or "").strip()
    base_model_path = Path(base_model_value) if base_model_value else None
    stats_path = (
        openvla_dataset_statistics_path(base_model_path, policy_type)
        if base_model_path is not None
        else None
    )
    checkpoint_path = Path(checkpoint_value) if checkpoint_value else None
    weights_path = checkpoint_path / "actor" / "model_state_dict" / "full_weights.pt" if checkpoint_path is not None else None

    base_status = _dependency_path_status(base_model_path, want_file=False)
    stats_status = _dependency_path_status(stats_path, want_file=True)
    checkpoint_status = _dependency_path_status(checkpoint_path, want_file=False)
    weights_status = _dependency_path_status(weights_path, want_file=True)
    unnorm_status = _dependency_unnorm_status(stats_path, unnorm_key)
    errors = _openvla_dependency_errors(config)
    return {
        "base_model": base_status,
        "stats_json": stats_status,
        "unnorm_key": unnorm_status,
        "checkpoint": checkpoint_status,
        "weights": weights_status,
        "status": "ok" if not errors else f"missing: {'; '.join(errors)}",
    }


def _dependency_path_status(path: Path | None, *, want_file: bool) -> str:
    if path is None:
        return "missing"
    exists = path.is_file() if want_file else path.exists()
    return f"ok {path}" if exists else f"missing {path}"


def _dependency_unnorm_status(stats_path: Path | None, unnorm_key: str) -> str:
    if not unnorm_key:
        return "missing"
    if stats_path is None or not stats_path.is_file():
        return f"unknown {unnorm_key}"
    try:
        statistics = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"unreadable {unnorm_key}"
    if isinstance(statistics, dict) and unnorm_key in statistics:
        return f"ok {unnorm_key}"
    return f"missing {unnorm_key}"


def _print_openvla_dependency_status(config: dict[str, Any]) -> None:
    if _policy_type(config) not in OPENVLA_POLICY_TYPES:
        return
    policy = config.get("policy", {})
    base_model_path = str(policy.get("base_model_path") or "")
    checkpoint_path = str(policy.get("checkpoint_path") or "")
    unnorm_key = str(policy.get("unnorm_key") or "")
    stats_path = (
        str(openvla_dataset_statistics_path(Path(base_model_path), _policy_type(config)))
        if base_model_path
        else ""
    )
    print(f"resolved_base_model_path: {base_model_path}")
    print(f"resolved_dataset_statistics_path: {stats_path}")
    print(f"resolved_checkpoint_path: {checkpoint_path}")
    print(f"resolved_unnorm_key: {unnorm_key}")
    dependency_errors = _openvla_dependency_errors(config)
    if dependency_errors:
        print(f"openvla_dependency_status: missing ({'; '.join(dependency_errors)})")
    else:
        print("openvla_dependency_status: ok")
