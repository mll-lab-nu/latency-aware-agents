from __future__ import annotations

from typing import Any


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
    except ImportError:
        return

    if getattr(ActorState.record_env_step, "_latency_bench_atari_episode_stats_patched", False):
        return

    original_record_env_step = ActorState.record_env_step

    def record_env_step_full_game(self, reward, terminated: bool, truncated: bool, info, rollout_step):
        done = bool(terminated or truncated)
        true_episode_end = is_true_atari_episode_end(terminated, truncated, info)
        if not done or true_episode_end:
            return original_record_env_step(self, reward, terminated, truncated, info, rollout_step)

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


def _batched_info_for_agent(infos: Any, agent_idx: int) -> dict[str, Any]:
    if isinstance(infos, (list, tuple)):
        if agent_idx < len(infos) and isinstance(infos[agent_idx], dict):
            return infos[agent_idx]
        return {}

    if isinstance(infos, dict):
        agent_info: dict[str, Any] = {}
        for key in (TRUE_EPISODE_END_INFO_KEY, "game_over", "lives", "TimeLimit.truncated"):
            if key in infos:
                value = _info_value_for_agent(infos[key], agent_idx)
                if value is not None:
                    agent_info[key] = value
        return agent_info

    return {}


def _is_true_batched_episode_end(info: dict[str, Any]) -> bool:
    if TRUE_EPISODE_END_INFO_KEY in info:
        return bool(info[TRUE_EPISODE_END_INFO_KEY])
    truncated = bool(info.get("TimeLimit.truncated", False))
    return is_true_atari_episode_end(terminated=True, truncated=truncated, info=info)


def _patch_batched_sample_factory_atari_episode_stats() -> None:
    try:
        from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
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

        return original_process_env_step(self, rewards, report_dones, infos)

    process_env_step_full_game._latency_bench_atari_episode_stats_patched = True
    process_env_step_full_game._latency_bench_original_process_env_step = original_process_env_step
    BatchedVectorEnvRunner._process_env_step = process_env_step_full_game
