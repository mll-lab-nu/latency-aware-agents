"""MyoChallenge Table Tennis P2 Gymnasium integration.

The MyoSuite package is imported from the Gymnasium entry point rather than at
module import time.  This keeps the rest of the benchmark importable when the
optional musculoskeletal simulator is not installed.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from latency_bench.envs.tasks.myosuite import MyoSuiteEnv


GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID = "LatencyBench/MyoChallengeTableTennisP2-v0"
MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID = "myoChallengeTableTennisP2-v0"
MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA = "default"
MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE = (224, 224)


class MyoChallengeTableTennisP2Env(MyoSuiteEnv):
    """Expose MyoSuite's native P2 state, action, score, and RGB interfaces."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        *,
        base_env_id: str = MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID,
        render_mode: str = "rgb_array",
        base_make_kwargs: dict[str, Any],
        camera_name: str = MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA,
        render_width: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[0],
        render_height: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[1],
    ) -> None:
        # MyoSuite registers its environments during import.  It is deliberately
        # imported by the factory below, not by this module.
        super().__init__(
            base_env_id=base_env_id,
            render_mode=render_mode,
            base_make_kwargs=base_make_kwargs,
            camera_name=camera_name,
            render_width=render_width,
            render_height=render_height,
        )

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = super().step(action)
        info = dict(info)
        info["solved"] = bool(info["solved"])
        info["score"] = float(info["solved"])
        return observation, reward, terminated, truncated, info


def make_latency_sensitive_myochallenge_table_tennis_p2_env(
    *,
    base_env_id: str = MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID,
    render_mode: str = "rgb_array",
    base_make_kwargs: dict[str, Any],
    camera_name: str = MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA,
    render_width: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[0],
    render_height: int = MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[1],
) -> MyoChallengeTableTennisP2Env:
    """Construct P2 after lazily importing MyoSuite's environment registry."""
    import myosuite  # noqa: F401

    return MyoChallengeTableTennisP2Env(
        base_env_id=base_env_id,
        render_mode=render_mode,
        base_make_kwargs=base_make_kwargs,
        camera_name=camera_name,
        render_width=render_width,
        render_height=render_height,
    )


gym.register(
    id=GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID,
    entry_point=make_latency_sensitive_myochallenge_table_tennis_p2_env,
    kwargs={
        "base_env_id": MYOCHALLENGE_TABLE_TENNIS_P2_BASE_ENV_ID,
        "render_mode": "rgb_array",
        "base_make_kwargs": {},
        "camera_name": MYOCHALLENGE_TABLE_TENNIS_P2_CAMERA,
        "render_width": MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[0],
        "render_height": MYOCHALLENGE_TABLE_TENNIS_P2_RENDER_SIZE[1],
    },
)
