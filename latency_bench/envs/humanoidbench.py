"""Register HumanoidBench with its MuJoCo 3.3 named-index compatibility patch."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import humanoid_bench  # noqa: F401
import numpy as np
from humanoid_bench.dmc_deps import dmc_sizes


H1HAND_BALANCE_HARD_ENV_ID = "LatencyBench/H1HandBalanceHard-v0"
HUMANOIDBENCH_H1HAND_BALANCE_HARD_ENV_ID = "h1hand-balance_hard-v0"


# HumanoidBench's generated table targets MuJoCo 3.1.6. These fields were
# removed or renamed in MuJoCo 3.3 before the named indexers are constructed.
for _struct_name, _field_names in {
    "mjmodel": ("flex_xvert0", "tex_rgb"),
    "mjdata": ("qLDiagSqrtInv",),
}.items():
    for _field_name in _field_names:
        del dmc_sizes.array_sizes[_struct_name][_field_name]


class H1HandBalanceHardEnv(gym.Wrapper):
    """Expose HumanoidBench's H1Hand task with its required ndarray action type."""

    def __init__(self, *, render_mode: str, width: int, height: int) -> None:
        env = gym.make(
            HUMANOIDBENCH_H1HAND_BALANCE_HARD_ENV_ID,
            render_mode=render_mode,
            width=width,
            height=height,
        )
        super().__init__(env)

    def step(self, action: Any):
        return self.env.step(np.asarray(action, dtype=self.env.action_space.dtype))


def make_h1hand_balance_hard_env(
    *,
    render_mode: str,
    width: int,
    height: int,
) -> H1HandBalanceHardEnv:
    return H1HandBalanceHardEnv(
        render_mode=render_mode,
        width=width,
        height=height,
    )


gym.register(
    id=H1HAND_BALANCE_HARD_ENV_ID,
    entry_point=make_h1hand_balance_hard_env,
    max_episode_steps=1000,
)
