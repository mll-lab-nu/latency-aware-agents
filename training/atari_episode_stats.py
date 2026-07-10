from __future__ import annotations

import numbers
from typing import Any

import numpy as np

from latency_bench.envs import atari


def _info_value_for_agent(value: Any, agent_idx: int) -> Any:
    if hasattr(value, "numel") and hasattr(value, "shape"):
        if value.numel() == 1:
            return value.item()
        if len(value.shape) >= 1 and len(value) > agent_idx:
            item = value[agent_idx]
            return item.item() if hasattr(item, "item") else item
        return None

    if isinstance(value, (list, tuple)) and len(value) > agent_idx:
        return value[agent_idx]

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        if len(value.shape) >= 1 and len(value) > agent_idx:
            item = value[agent_idx]
            return item.item() if hasattr(item, "item") else item
        return None

    return value


def _episode_extra_stats_for_agent(value: Any, agent_idx: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    agent_stats: dict[str, Any] = {}
    for key, stat_value in value.items():
        key_str = str(key)
        if isinstance(stat_value, dict):
            nested_stats = _episode_extra_stats_for_agent(stat_value, agent_idx)
            for nested_key, nested_value in nested_stats.items():
                agent_stats[f"{key_str}/{nested_key}"] = nested_value
            continue

        agent_value = _info_value_for_agent(stat_value, agent_idx)
        if agent_value is not None:
            agent_stats[key_str] = agent_value

    return agent_stats


def batched_info_for_agent(infos: Any, agent_idx: int) -> dict[str, Any]:
    if isinstance(infos, (list, tuple)):
        if agent_idx < len(infos) and isinstance(infos[agent_idx], dict):
            return infos[agent_idx]
        return {}

    if isinstance(infos, dict):
        agent_info: dict[str, Any] = {}
        for key in (
            atari.TRUE_EPISODE_END_INFO_KEY,
            "game_over",
            "lives",
            "TimeLimit.truncated",
            "episodic_return",
            "episodic_raw_return",
        ):
            if key in infos:
                value = _info_value_for_agent(infos[key], agent_idx)
                if value is not None:
                    agent_info[key] = value
        if "episode_extra_stats" in infos:
            extra_stats = _episode_extra_stats_for_agent(infos["episode_extra_stats"], agent_idx)
            if extra_stats:
                agent_info["episode_extra_stats"] = extra_stats
        return agent_info

    return {}


def _numeric_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Number):
        return float(value)
    if hasattr(value, "numel") and callable(value.numel) and value.numel() == 1:
        try:
            item = value.item()
        except (TypeError, ValueError):
            return None
        if isinstance(item, bool):
            return None
        if isinstance(item, numbers.Number):
            return float(item)
    return None


def flatten_numeric_episode_extra_stats(value: Any, prefix: str = "") -> dict[str, float]:
    if not isinstance(value, dict):
        numeric_value = _numeric_scalar(value)
        if prefix and numeric_value is not None:
            return {prefix: numeric_value}
        return {}

    flattened: dict[str, float] = {}
    for key, item in value.items():
        key_str = str(key)
        next_prefix = f"{prefix}/{key_str}" if prefix else key_str
        flattened.update(flatten_numeric_episode_extra_stats(item, next_prefix))
    return flattened


def raw_full_game_return_from_info(info: Any) -> float | None:
    if not isinstance(info, dict):
        return None

    for key in ("episodic_raw_return", "episodic_return"):
        value = _numeric_scalar(info.get(key))
        if value is not None:
            return value

    extra_stats = flatten_numeric_episode_extra_stats(info.get("episode_extra_stats") or {})
    for key in ("episode/raw_return", "episode/return", "raw_return", "return"):
        value = extra_stats.get(key)
        if value is not None:
            return float(value)
    return None


def apply_raw_full_game_return_to_report(report: Any, info: Any, episodic_key: str) -> None:
    if not isinstance(report, dict):
        return
    episodic_stats = report.get(episodic_key)
    if not isinstance(episodic_stats, dict):
        return
    raw_return = raw_full_game_return_from_info(info)
    if raw_return is not None:
        episodic_stats["reward"] = float(raw_return)


def episode_extra_stats_for_finished_agents(infos: Any, finished_agents: list[int]) -> dict[str, np.ndarray]:
    collected: dict[str, list[float]] = {}
    for agent_idx in finished_agents:
        info = batched_info_for_agent(infos, int(agent_idx))
        extra_stats = flatten_numeric_episode_extra_stats(info.get("episode_extra_stats") or {})
        for key, value in extra_stats.items():
            collected.setdefault(key, []).append(value)
    return {key: np.asarray(values, dtype=np.float64) for key, values in collected.items()}


def raw_full_game_returns_for_finished_agents(infos: Any, finished_agents: list[int]) -> np.ndarray | None:
    raw_returns: list[float] = []
    for agent_idx in finished_agents:
        raw_return = raw_full_game_return_from_info(batched_info_for_agent(infos, int(agent_idx)))
        if raw_return is None:
            return None
        raw_returns.append(float(raw_return))
    return np.asarray(raw_returns, dtype=np.float64)


def is_true_batched_episode_end(info: dict[str, Any]) -> bool:
    truncated = info.get("TimeLimit.truncated", False)
    return atari.is_true_atari_episode_end(terminated=True, truncated=truncated, info=info)
