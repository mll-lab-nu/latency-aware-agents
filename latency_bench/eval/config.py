"""Evaluation config interpretation and CLI runtime overrides.

This module depends on config shape rather than executors or orchestration. It is
shared by the CLI, evaluation drivers, and reporting.
"""
from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from typing import Any

_SAMPLE_FACTORY_POLICY_TYPES = {
    "flappy_sf",
    "demon_attack_sf",
    "deadly_corridor_sf",
    "gymnasium_sf",
}


def resolve_evaluation_config(config: dict[str, Any]) -> None:
    """Materialize existing evaluation defaults at direct-dict and CLI boundaries.

    Serial evaluation, in-process inference and disabled pipeline profiling are
    the existing defaults. Worker capacity is explicitly optional. Experiment
    seed stays untouched: policy and suite RNGs distinguish omission from zero.
    Explicit filesystem inputs are expanded against the caller's working
    directory; model identifiers and paths embedded in model configs are not
    interpreted here.
    """
    defaults = {
        "evaluation": {"eval_parallel_envs": 1},
        "executor": {
            "simulated_inference_pool": False,
            "simulated_worker_capacity": None,
        },
        "logging": {"simulated_pipeline_profile": False},
    }
    for section, fields in defaults.items():
        for name, value in fields.items():
            if name not in config[section]:
                config[section][name] = value
    evaluation = config["evaluation"]
    for field in ("eval_max_steps", "eval_episodes"):
        if field in evaluation and evaluation[field] is not None:
            evaluation[field] = int(evaluation[field])
    evaluation["eval_parallel_envs"] = int(evaluation["eval_parallel_envs"])
    evaluation["eval_latency_values"] = _eval_latency_values(config)

    path_fields = {
        "policy": (
            "checkpoint_path",
            "model_config_path",
            "task_contract_path",
            "task_manifest_path",
            "latency_prompt_map_path",
            "backbone_path",
        ),
        "latency": ("profile_path",),
        "logging": ("output_dir",),
    }
    # HF snapshot filenames and sibling metadata depend on preserving symlinks.
    for section, fields in path_fields.items():
        if section not in config:
            continue
        for field in fields:
            if field in config[section] and config[section][field] not in (None, ""):
                config[section][field] = os.path.abspath(Path(config[section][field]).expanduser())


def apply_runtime_overrides(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    checkpoint_kind: str | None = None,
    latency_prompt_map_path: str | Path | None = None,
    latency_raw_frames: int | None = None,
    eval_max_steps: int | None = None,
    eval_parallel_envs: int | None = None,
    model_config_path: str | Path | None = None,
    task_contract_path: str | Path | None = None,
    task_manifest_path: str | Path | None = None,
    latency_method: str | None = None,
    latency_profile_path: str | Path | None = None,
    latency_seed: int | None = None,
) -> dict[str, Any]:
    if checkpoint_path is not None:
        config["policy"]["checkpoint_path"] = str(checkpoint_path)
    if model_config_path is not None:
        config["policy"]["model_config_path"] = str(model_config_path)
    if task_contract_path is not None:
        config["policy"]["task_contract_path"] = str(task_contract_path)
    if task_manifest_path is not None:
        config["policy"]["task_manifest_path"] = str(task_manifest_path)
    if latency_method is not None:
        config["latency"]["method"] = latency_method
    if latency_profile_path is not None:
        config["latency"]["profile_path"] = str(latency_profile_path)
    if latency_seed is not None:
        config["latency"]["seed"] = latency_seed
    if run_name is not None:
        config["experiment"]["name"] = run_name
    if output_dir is not None:
        config["logging"]["output_dir"] = str(output_dir)
    if checkpoint_kind is not None:
        config["policy"]["checkpoint_kind"] = checkpoint_kind
        if _is_sample_factory_policy(config):
            from training.common.config_adapter import apply_checkpoint_kind_override

            apply_checkpoint_kind_override(config["backend"], checkpoint_kind)
    if latency_prompt_map_path is not None:
        config["policy"]["latency_prompt_map_path"] = str(latency_prompt_map_path)
    if latency_raw_frames is not None:
        latency_ms = _latency_steps_to_ms(config, latency_raw_frames)
        _apply_eval_latency(config, latency_raw_frames, latency_ms)
        config["policy"]["latency_prompt_key"] = latency_raw_frames
        if "eval_suites" in config["evaluation"]:
            config["evaluation"]["eval_suites"] = {
                "fixed": [[latency_raw_frames]],
                "normal": [],
                "uniform": [],
            }
        if "eval_latency_values" in config["evaluation"]:
            config["evaluation"]["eval_latency_values"] = [latency_raw_frames]
    if eval_max_steps is not None:
        config["evaluation"]["eval_max_steps"] = eval_max_steps
    if eval_parallel_envs is not None:
        config["evaluation"]["eval_parallel_envs"] = eval_parallel_envs
    return config


def _is_sample_factory_policy(config: dict[str, Any]) -> bool:
    policy_type = config["policy"]["type"].strip().lower()
    return policy_type in _SAMPLE_FACTORY_POLICY_TYPES


def _evaluation_seed(config: dict[str, Any]) -> int:
    """Episode and reporting seed; omitted policy/suite RNG seeds remain optional."""
    return int(config["experiment"].get("seed", 0))


def _episode_seed(config: dict[str, Any], episode_id: int) -> int:
    seed = _evaluation_seed(config)
    if "episode_seed_offset" in config["evaluation"]:
        return seed + config["evaluation"]["episode_seed_offset"] + episode_id
    if _is_sample_factory_policy(config):
        return seed + 104729 + episode_id
    return seed + episode_id


def _eval_max_steps(config: dict[str, Any]) -> int:
    evaluation_cfg = config.get("evaluation")
    if isinstance(evaluation_cfg, dict) and evaluation_cfg.get("eval_max_steps") is not None:
        return int(evaluation_cfg["eval_max_steps"])
    raise ValueError("evaluation.eval_max_steps is required for latency_bench episode limits")


def _eval_episodes(config: dict[str, Any]) -> int:
    return config["evaluation"]["eval_episodes"]


def _selected_episode_count(config: dict[str, Any], episode_ids: Sequence[int] | None) -> int:
    if episode_ids is None:
        return _eval_episodes(config)
    return len(episode_ids)


def _eval_latency_values(config: dict[str, Any]) -> list[int] | None:
    evaluation_cfg = config.get("evaluation")
    if not isinstance(evaluation_cfg, dict) or "eval_latency_values" not in evaluation_cfg:
        return None
    values = evaluation_cfg["eval_latency_values"]
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        values = [item.strip() for item in str(values).split(",") if item.strip()]
    elif not isinstance(values, Sequence):
        values = [values]
    return [int(value) for value in values]


def _latency_steps_from_config(config: dict[str, Any]) -> int | None:
    executor_mode = config["executor"]["mode"].strip().lower()
    if executor_mode == "realtime":
        return None
    latency_cfg = config["latency"]
    if latency_cfg["method"] == "zero":
        return 0

    if latency_cfg["method"] != "fixed":
        return None
    latency_steps = (
        float(latency_cfg["fixed_latency_ms"])
        * float(config["env"]["env_fps"])
        / 1000.0
    )
    rounded_steps = round(latency_steps)
    if abs(latency_steps - rounded_steps) > 1e-9:
        return None
    return int(rounded_steps)


def _latency_steps_to_ms(config: dict[str, Any], latency_steps: int) -> float:
    env_fps = float(config["env"]["env_fps"])
    return latency_steps * 1000.0 / env_fps


def _apply_eval_latency(config: dict[str, Any], latency_steps: int, latency_ms: float) -> None:
    latency_cfg = config["latency"]
    if latency_steps == 0:
        latency_cfg["method"] = "zero"
        latency_cfg.pop("fixed_latency_ms", None)
        return
    latency_cfg["method"] = "fixed"
    latency_cfg["fixed_latency_ms"] = latency_ms
