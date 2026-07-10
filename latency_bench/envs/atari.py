from __future__ import annotations

from typing import Any


TRUE_EPISODE_END_INFO_KEY = "is_true_episode_end"


def is_true_atari_episode_end(
    terminated: bool,
    truncated: bool,
    info: dict[str, Any] | None,
) -> bool:
    """Distinguish true Atari game ends from life-loss pseudo terminals."""

    if truncated:
        return True
    if not terminated:
        return False
    if info is None:
        return True
    if TRUE_EPISODE_END_INFO_KEY in info:
        return info[TRUE_EPISODE_END_INFO_KEY]
    if "game_over" in info:
        return info["game_over"]
    if "lives" in info:
        return info["lives"] <= 0
    return True
