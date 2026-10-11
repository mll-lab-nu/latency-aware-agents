"""Evaluation config interpretation and CLI runtime overrides.

This module depends on config shape rather than executors or orchestration. It is
shared by the CLI, evaluation drivers, and reporting.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from latency_bench.policy.model_source import set_model_files



def apply_runtime_overrides(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    checkpoint_kind: str | None = None,
    latency_prompt_map_path: str | Path | None = None,
    latency_raw_frames: int | None = None,
    max_env_steps: int | None = None,
    parallel_envs: int | None = None,
    model_config_path: str | Path | None = None,
    task_contract_path: str | Path | None = None,
    task_manifest_path: str | Path | None = None,
    latency_method: str | None = None,
    latency_profile_path: str | Path | None = None,
    latency_seed: int | None = None,
) -> dict[str, Any]:
    if checkpoint_path is not None or model_config_path is not None:
        set_model_files(config, checkpoint=checkpoint_path, model_config=model_config_path)
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
        config["evaluation"]["suites"] = []
    if max_env_steps is not None:
        config["evaluation"]["max_env_steps"] = max_env_steps
    if parallel_envs is not None:
        config["executor"]["parallel_envs"] = parallel_envs
    return config


def _is_sample_factory_policy(config: dict[str, Any]) -> bool:
    policy_type = config["policy"]["type"].strip().lower()
    return policy_type == "sample_factory"


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


def _selected_episode_count(config: dict[str, Any], episode_ids: Sequence[int] | None) -> int:
    if episode_ids is None:
        return config["evaluation"]["episodes"]
    return len(episode_ids)


def _latency_steps_from_config(config: dict[str, Any]) -> int | None:
    executor_mode = config["executor"]["mode"].strip().lower()
    if executor_mode == "realtime":
        return None
    if executor_mode == "paused":
        return 0
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
