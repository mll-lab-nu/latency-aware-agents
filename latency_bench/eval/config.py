"""Evaluation config interpretation and CLI runtime overrides.

This module depends on config shape rather than executors or orchestration. It is
shared by the CLI, evaluation drivers, and reporting.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

_SAMPLE_FACTORY_POLICY_TYPES = {
    "flappy_sf",
    "demon_attack_sf",
    "deadly_corridor_sf",
    "gymnasium_sf",
}


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
        checkpoint_key = (
            "model_path"
            if config["policy"]["type"] == "official_gr00t"
            else "checkpoint_path"
        )
        config["policy"][checkpoint_key] = str(checkpoint_path)
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
        config["experiment"]["name"] = str(run_name)
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
        latency_ms = (
            int(latency_raw_frames)
            * 1000.0
            / float(config["env"]["env_fps"])
        )
        _apply_eval_latency(config, int(latency_raw_frames), latency_ms)
        config["policy"]["latency_prompt_key"] = int(latency_raw_frames)
        if "eval_suites" in config["evaluation"]:
            config["evaluation"]["eval_suites"] = {
                "fixed": [[int(latency_raw_frames)]],
                "normal": [],
                "uniform": [],
            }
        if "eval_latency_values" in config["evaluation"]:
            config["evaluation"]["eval_latency_values"] = [
                int(latency_raw_frames)
            ]
    if eval_max_steps is not None:
        config["evaluation"]["eval_max_steps"] = int(eval_max_steps)
    if eval_parallel_envs is not None:
        config["evaluation"]["eval_parallel_envs"] = int(eval_parallel_envs)
    return config


def _is_sample_factory_policy(config: dict[str, Any]) -> bool:
    policy_type = str(config.get("policy", {}).get("type", "")).strip().lower()
    return policy_type in _SAMPLE_FACTORY_POLICY_TYPES


def _episode_seed(config: dict[str, Any], episode_id: int) -> int:
    seed = int(config["experiment"].get("seed", 0))
    if _is_sample_factory_policy(config):
        return seed + 104729 + int(episode_id)
    return seed + int(episode_id)


def _eval_max_steps(config: dict[str, Any]) -> int:
    evaluation_cfg = config.get("evaluation")
    if isinstance(evaluation_cfg, dict) and evaluation_cfg.get("eval_max_steps") is not None:
        return int(evaluation_cfg["eval_max_steps"])
    raise ValueError("evaluation.eval_max_steps is required for latency_bench episode limits")


def _eval_episodes(config: dict[str, Any]) -> int:
    return int(config["evaluation"]["eval_episodes"])


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
    executor_mode = str(config.get("executor", {}).get("mode", "")).strip().lower()
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


def _infer_latency_steps(config: dict[str, Any]) -> int | None:
    return _latency_steps_from_config(config)


def _train_latency_steps(config: dict[str, Any]) -> int | None:
    return _latency_steps_from_config(config)


def _latency_steps_to_ms(config: dict[str, Any], latency_steps: int) -> float:
    env_fps = float(config["env"]["env_fps"])
    if env_fps <= 0:
        raise ValueError("env.env_fps must be positive")
    return int(latency_steps) * 1000.0 / env_fps


def _apply_eval_latency(config: dict[str, Any], latency_steps: int, latency_ms: float) -> None:
    latency_cfg = config.setdefault("latency", {})
    if int(latency_steps) == 0:
        latency_cfg["method"] = "zero"
        latency_cfg.pop("fixed_latency_ms", None)
        return
    latency_cfg["method"] = "fixed"
    latency_cfg["fixed_latency_ms"] = float(latency_ms)
