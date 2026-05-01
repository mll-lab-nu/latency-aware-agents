from __future__ import annotations

import numbers
from typing import Any

import numpy as np


TRUE_EPISODE_END_INFO_KEY = "is_true_episode_end"


def is_true_atari_episode_end(terminated: bool, truncated: bool, info: dict[str, Any] | None) -> bool:
    """Distinguish true Atari game ends from life-loss pseudo terminals."""

    if bool(truncated):
        return True
    if not bool(terminated):
        return False

    info = info or {}
    if TRUE_EPISODE_END_INFO_KEY in info:
        return bool(info[TRUE_EPISODE_END_INFO_KEY])
    if "game_over" in info:
        return bool(info["game_over"])
    if "lives" in info:
        try:
            return int(info["lives"]) <= 0
        except (TypeError, ValueError):
            return True
    return True


def patch_sample_factory_atari_episode_stats() -> None:
    """Patch Sample Factory to report Atari episode stats only on full games."""

    _patch_non_batched_sample_factory_atari_episode_stats()
    _patch_batched_sample_factory_atari_episode_stats()


def _patch_non_batched_sample_factory_atari_episode_stats() -> None:
    try:
        from sample_factory.algo.sampling.non_batched_sampling import ActorState
        from sample_factory.algo.utils.misc import EPISODIC
    except ImportError:
        return

    if getattr(ActorState.record_env_step, "_latency_bench_atari_episode_stats_patched", False):
        return

    original_record_env_step = ActorState.record_env_step

    def record_env_step_full_game(self, reward, terminated: bool, truncated: bool, info, rollout_step):
        done = bool(terminated or truncated)
        true_episode_end = is_true_atari_episode_end(terminated, truncated, info)
        if not done or true_episode_end:
            report = original_record_env_step(self, reward, terminated, truncated, info, rollout_step)
            if done and true_episode_end:
                _apply_raw_full_game_return_to_report(report, info, EPISODIC)
                _filter_report_clipped_return_stats(report, EPISODIC)
            return report

        episode_reward = self.last_episode_reward
        duration_before_step = self.last_episode_duration
        original_record_env_step(self, reward, terminated, truncated, info, rollout_step)

        duration_increment = self.env_info.frameskip if self.cfg.summaries_use_frameskip else 1
        self.last_episode_reward = episode_reward
        self.last_episode_duration = duration_before_step + duration_increment
        return None

    record_env_step_full_game._latency_bench_atari_episode_stats_patched = True
    record_env_step_full_game._latency_bench_original_record_env_step = original_record_env_step
    ActorState.record_env_step = record_env_step_full_game


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


def _batched_info_for_agent(infos: Any, agent_idx: int) -> dict[str, Any]:
    if isinstance(infos, (list, tuple)):
        if agent_idx < len(infos) and isinstance(infos[agent_idx], dict):
            return infos[agent_idx]
        return {}

    if isinstance(infos, dict):
        agent_info: dict[str, Any] = {}
        for key in (
            TRUE_EPISODE_END_INFO_KEY,
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


def _flatten_numeric_episode_extra_stats(value: Any, prefix: str = "") -> dict[str, float]:
    if not isinstance(value, dict):
        numeric_value = _numeric_scalar(value)
        if prefix and numeric_value is not None:
            return {prefix: numeric_value}
        return {}

    flattened: dict[str, float] = {}
    for key, item in value.items():
        key_str = str(key)
        next_prefix = f"{prefix}/{key_str}" if prefix else key_str
        flattened.update(_flatten_numeric_episode_extra_stats(item, next_prefix))
    return flattened


def _is_clipped_return_metric_key(key: str) -> bool:
    return "clipped_return" in str(key)


def _filter_episode_extra_stats_for_reporting(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    filtered: dict[str, Any] = {}
    for key, item in value.items():
        key_str = str(key)
        if _is_clipped_return_metric_key(key_str):
            continue
        filtered[key_str] = _filter_episode_extra_stats_for_reporting(item)
    return filtered


def _raw_full_game_return_from_info(info: Any) -> float | None:
    if not isinstance(info, dict):
        return None

    for key in ("episodic_raw_return", "episodic_return"):
        value = _numeric_scalar(info.get(key))
        if value is not None:
            return value

    extra_stats = _flatten_numeric_episode_extra_stats(info.get("episode_extra_stats") or {})
    for key in ("episode/raw_return", "episode/return", "raw_return", "return"):
        value = extra_stats.get(key)
        if value is not None:
            return float(value)
    return None


def _apply_raw_full_game_return_to_report(report: Any, info: Any, episodic_key: str) -> None:
    if not isinstance(report, dict):
        return
    episodic_stats = report.get(episodic_key)
    if not isinstance(episodic_stats, dict):
        return
    raw_return = _raw_full_game_return_from_info(info)
    if raw_return is not None:
        episodic_stats["reward"] = float(raw_return)


def _filter_report_clipped_return_stats(report: Any, episodic_key: str) -> None:
    if not isinstance(report, dict):
        return
    episodic_stats = report.get(episodic_key)
    if not isinstance(episodic_stats, dict):
        return

    for key in list(episodic_stats):
        if _is_clipped_return_metric_key(str(key)):
            del episodic_stats[key]

    if "episode_extra_stats" in episodic_stats:
        episodic_stats["episode_extra_stats"] = _filter_episode_extra_stats_for_reporting(
            episodic_stats["episode_extra_stats"]
        )


def _episode_extra_stats_for_finished_agents(infos: Any, finished_agents: list[int]) -> dict[str, np.ndarray]:
    collected: dict[str, list[float]] = {}
    for agent_idx in finished_agents:
        info = _batched_info_for_agent(infos, int(agent_idx))
        extra_stats = _flatten_numeric_episode_extra_stats(info.get("episode_extra_stats") or {})
        for key, value in extra_stats.items():
            if _is_clipped_return_metric_key(key):
                continue
            collected.setdefault(key, []).append(value)
    return {key: np.asarray(values, dtype=np.float64) for key, values in collected.items()}


def _raw_full_game_returns_for_finished_agents(infos: Any, finished_agents: list[int]) -> np.ndarray | None:
    raw_returns: list[float] = []
    for agent_idx in finished_agents:
        raw_return = _raw_full_game_return_from_info(_batched_info_for_agent(infos, int(agent_idx)))
        if raw_return is None:
            return None
        raw_returns.append(float(raw_return))
    return np.asarray(raw_returns, dtype=np.float64)


def _is_true_batched_episode_end(info: dict[str, Any]) -> bool:
    if TRUE_EPISODE_END_INFO_KEY in info:
        return bool(info[TRUE_EPISODE_END_INFO_KEY])
    truncated = bool(info.get("TimeLimit.truncated", False))
    return is_true_atari_episode_end(terminated=True, truncated=truncated, info=info)


def _patch_batched_sample_factory_atari_episode_stats() -> None:
    try:
        from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
        from sample_factory.algo.utils.misc import EPISODIC
    except ImportError:
        return

    if getattr(
        BatchedVectorEnvRunner._process_env_step,
        "_latency_bench_atari_episode_stats_patched",
        False,
    ):
        return

    original_process_env_step = BatchedVectorEnvRunner._process_env_step

    def process_env_step_full_game(self, rewards, dones_orig, infos):
        dones_cpu = dones_orig.cpu()
        if dones_cpu.sum().item() <= 0:
            return original_process_env_step(self, rewards, dones_orig, infos)

        report_dones = dones_orig.clone()
        for agent_idx in dones_cpu.nonzero(as_tuple=True)[0].tolist():
            info = _batched_info_for_agent(infos, int(agent_idx))
            if not _is_true_batched_episode_end(info):
                report_dones[agent_idx] = False

        reports = original_process_env_step(self, rewards, report_dones, infos)
        if reports:
            finished_agents = [int(agent_idx) for agent_idx in report_dones.cpu().nonzero(as_tuple=True)[0].tolist()]
            raw_returns = _raw_full_game_returns_for_finished_agents(infos, finished_agents)
            extra_stats = _episode_extra_stats_for_finished_agents(infos, finished_agents)
            for report in reports:
                episodic_stats = report.get(EPISODIC)
                if isinstance(episodic_stats, dict):
                    if raw_returns is not None:
                        episodic_stats["reward"] = raw_returns
                    if extra_stats:
                        episodic_stats.update(extra_stats)
                    _filter_report_clipped_return_stats(report, EPISODIC)
        return reports

    process_env_step_full_game._latency_bench_atari_episode_stats_patched = True
    process_env_step_full_game._latency_bench_original_process_env_step = original_process_env_step
    BatchedVectorEnvRunner._process_env_step = process_env_step_full_game
