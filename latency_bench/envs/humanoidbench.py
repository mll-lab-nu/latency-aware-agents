"""Register the HumanoidBench task family and its MuJoCo compatibility patch."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import humanoid_bench
import numpy as np
from humanoid_bench.dmc_deps import dmc_sizes


# HumanoidBench's generated table targets MuJoCo 3.1.6. These fields were
# removed or renamed in MuJoCo 3.3 before the named indexers are constructed.
for _struct_name, _field_names in {
    "mjmodel": ("flex_xvert0", "tex_rgb"),
    "mjdata": ("qLDiagSqrtInv",),
}.items():
    for _field_name in _field_names:
        del dmc_sizes.array_sizes[_struct_name][_field_name]


HUMANOIDBENCH_ENV_SPECS = {
    "h1hand_balance_hard": {
        "env_id": "LatencyBench/H1HandBalanceHard-v0",
        "native_env_id": "h1hand-balance_hard-v0",
        "observation_shape": (177,),
        "max_episode_steps": gym.spec("h1hand-balance_hard-v0").max_episode_steps,
    },
    "h1hand_hurdle": {
        "env_id": "LatencyBench/H1HandHurdle-v0",
        "native_env_id": "h1hand-hurdle-v0",
        "observation_shape": (151,),
        "max_episode_steps": gym.spec("h1hand-hurdle-v0").max_episode_steps,
    },
    "h1hand_run": {
        "env_id": "LatencyBench/H1HandRun-v0",
        "native_env_id": "h1hand-run-v0",
        "observation_shape": (151,),
        "max_episode_steps": gym.spec("h1hand-run-v0").max_episode_steps,
    },
    "h1hand_basketball": {
        "env_id": "LatencyBench/H1HandBasketball-v0",
        "native_env_id": "h1hand-basketball-v0",
        "observation_shape": (164,),
        "max_episode_steps": gym.spec("h1hand-basketball-v0").max_episode_steps,
    },
    "h1strong_highbar_hard": {
        "env_id": "LatencyBench/H1StrongHighBarHard-v0",
        "native_env_id": "h1strong-highbar_hard-v0",
        "observation_shape": (151,),
        "max_episode_steps": gym.spec("h1strong-highbar_hard-v0").max_episode_steps,
    },
}


class HumanoidBenchEnv(gym.Wrapper):
    """Adapt one native HumanoidBench task to LatencyBench's action boundary."""

    metadata = humanoid_bench.env.HumanoidEnv.metadata

    def __init__(
        self,
        *,
        native_env_id: str,
        render_mode: str,
        width: int,
        height: int,
    ) -> None:
        env = gym.make(
            native_env_id,
            render_mode=render_mode,
            width=width,
            height=height,
        )
        super().__init__(env)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self.env.unwrapped.seed(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any):
        return self.env.step(np.asarray(action, dtype=self.env.action_space.dtype))


for _env_spec in HUMANOIDBENCH_ENV_SPECS.values():
    gym.register(
        id=_env_spec["env_id"],
        entry_point=HumanoidBenchEnv,
        kwargs={"native_env_id": _env_spec["native_env_id"]},
        max_episode_steps=_env_spec["max_episode_steps"],
    )
