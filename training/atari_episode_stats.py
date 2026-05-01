from __future__ import annotations

from typing import Any


def is_true_atari_episode_end(terminated: bool, truncated: bool, info: dict[str, Any] | None) -> bool:
    """Distinguish true Atari game ends from life-loss pseudo terminals."""

    if bool(truncated):
        return True
    if not bool(terminated):
        return False

    info = info or {}
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
