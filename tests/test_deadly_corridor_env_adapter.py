from __future__ import annotations

from collections import deque

import numpy as np

from latency_bench.envs.deadly_corridor import DeadlyCorridorEnvAdapter


class ScriptedVizdoomGame:
    def __init__(self):
        self.seeds: list[int] = []

    def set_seed(self, seed: int) -> None:
        self.seeds.append(int(seed))


class ScriptedVizdoomEnv:
    def __init__(self):
        self.curr_seed = 0
        self.game = ScriptedVizdoomGame()
        self.seed_calls: list[int] = []
        self.reset_seeds: list[int | None] = []

    @property
    def unwrapped(self):
        return self

    def seed(self, seed: int):
        self.seed_calls.append(int(seed))
        self.curr_seed = int(seed) + 1000
        return [self.curr_seed, None]

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return np.zeros((3, 4, 4), dtype=np.uint8), {"reset_seed": seed}


def test_deadly_corridor_reset_reseeds_initialized_vizdoom_game_before_episode():
    gym_env = ScriptedVizdoomEnv()
    adapter = object.__new__(DeadlyCorridorEnvAdapter)
    adapter.gym_env = gym_env
    adapter.env_fps = 60.0
    adapter.env_step = 3
    adapter._last_info = {}
    adapter._last_frame = None
    adapter._observed_frames = deque(maxlen=4)

    adapter.reset(seed=17)

    assert gym_env.seed_calls == [17]
    assert gym_env.game.seeds == [1017]
    assert gym_env.reset_seeds == [None]
