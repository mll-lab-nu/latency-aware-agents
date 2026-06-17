"""Pure interpretation of an eval config: seeds, episode counts, and latency math.

Leaf module — depends only on the config dict shape, never on executors or
orchestration. Shared by ``run.py`` (single-run + sweep orchestration) and
``eval_reporting.py``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_SAMPLE_FACTORY_POLICY_TYPES = {
    "flappy_sf",
    "demon_attack_sf",
    "deadly_corridor_sf",
}


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
    latency_cfg = config.get("latency", {})
    latency_type = str(latency_cfg.get("type", "")).strip().lower()
    if latency_type == "zero":
        return 0

    if "fixed_latency_ms" not in latency_cfg:
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
        latency_cfg["type"] = "zero"
        latency_cfg.pop("fixed_latency_ms", None)
        return
    latency_cfg["type"] = "fixed"
    latency_cfg["fixed_latency_ms"] = float(latency_ms)
