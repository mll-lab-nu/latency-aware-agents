from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from latency_bench.core.types import Action
from latency_bench.envs.deadly_corridor import (
    DeadlyCorridorEnvAdapter,
    DeadlyCorridorVlaEnvAdapter,
    _semantic_to_runtime_multibinary,
    _deadly_runtime_button_names,
    DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER,
)
from latency_bench.envs.raw_rgb import (
    ENV_RAW_RGB_FRAME_STACK_INFO_KEY,
    RawRgbFrameStackBuffer,
)


class ScriptedVizdoomGame:
    def __init__(self):
        self.seeds: list[int] = []
        self.screen_buffer = np.zeros((3, 4, 4), dtype=np.uint8)

    def get_available_buttons(self):
        return DEADLY_CORRIDOR_SEMANTIC_BUTTON_ORDER

    def set_seed(self, seed: int) -> None:
        self.seeds.append(int(seed))

    def get_state(self):
        return self


class ScriptedVizdoomEnv:
    def __init__(self, steps=None):
        self.curr_seed = 0
        self.game = ScriptedVizdoomGame()
        self.seed_calls: list[int] = []
        self.reset_seeds: list[int | None] = []
        self.steps = list(steps or [])

    @property
    def unwrapped(self):
        return self

    def seed(self, seed: int):
        self.seed_calls.append(int(seed))
        self.curr_seed = int(seed) + 1000
        return [self.curr_seed, None]

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.game.screen_buffer = np.ones((3, 4, 4), dtype=np.uint8)
        return self.game.screen_buffer.copy(), {"reset_seed": seed}

    def step(self, action):
        value = int(self.steps.pop(0))
        self.game.screen_buffer = np.full((3, 4, 4), value, dtype=np.uint8)
        return self.game.screen_buffer.copy(), 0.0, False, False, {"value": value}


def test_deadly_corridor_multibinary_buttons_follow_runtime_order():
    semantic = [1, 0, 1, 0, 1, 0, 1]
    runtime_order = [
        "ATTACK",
        "TURN_LEFT",
        "TURN_RIGHT",
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "MOVE_LEFT",
        "MOVE_RIGHT",
    ]

    assert _semantic_to_runtime_multibinary(semantic, runtime_order) == [1, 1, 0, 1, 0, 1, 0]


def test_deadly_corridor_reset_reseeds_initialized_vizdoom_game_before_episode():
    gym_env = ScriptedVizdoomEnv()
    adapter = object.__new__(DeadlyCorridorEnvAdapter)
    adapter.gym_env = gym_env
    adapter.env_fps = 60.0
    adapter.frame_stack = 4
    adapter.env_step = 3
    adapter.export_env_raw_rgb_frames = False
    adapter._last_info = {}
    adapter._last_frame = None
    adapter._observed_frames = deque(maxlen=4)
    adapter._raw_rgb_frames = RawRgbFrameStackBuffer(num_frames=4)

    adapter.reset(seed=17)

    assert gym_env.seed_calls == [17]
    assert gym_env.game.seeds == [1017]
    assert gym_env.reset_seeds == [None]


def test_deadly_corridor_observe_stacks_raw_frames_between_policy_observations():
    gym_env = ScriptedVizdoomEnv(steps=[2, 3])
    adapter = object.__new__(DeadlyCorridorEnvAdapter)
    adapter.gym_env = gym_env
    adapter.env_fps = 60.0
    adapter.frame_stack = 2
    adapter.env_step = 0
    adapter.export_env_raw_rgb_frames = True
    adapter._last_info = {}
    adapter._last_frame = None
    adapter._observed_frames = deque(maxlen=2)
    adapter._raw_rgb_frames = RawRgbFrameStackBuffer(num_frames=2)

    adapter.reset(seed=None)
    first_policy_observation = adapter.observe()
    adapter.step(Action(value=0, name="noop", is_noop=True))
    second_step = adapter.step(Action(value=0, name="noop", is_noop=True))
    second_policy_observation = adapter.observe()

    assert second_step.observation is not None
    assert first_policy_observation.data.shape == (6, 4, 4)
    assert second_policy_observation.data.shape == (6, 4, 4)
    assert np.all(second_policy_observation.data[:3] == 2)
    assert np.all(second_policy_observation.data[3:] == 3)
    raw_stack = second_policy_observation.metadata["env_raw_rgb_frame_stack"]
    assert raw_stack.shape == (2, 4, 4, 3)
    assert np.all(raw_stack[0] == 2)
    assert np.all(raw_stack[1] == 3)


def test_deadly_corridor_vla_observe_exports_raw_frame_stack():
    frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    adapter = object.__new__(DeadlyCorridorVlaEnvAdapter)
    adapter.env_fps = 35.0
    adapter.env_step = 0
    adapter.export_env_raw_rgb_frames = True
    adapter._last_info = {}
    adapter._last_frame = frame
    adapter._raw_rgb_frames = RawRgbFrameStackBuffer(num_frames=2)
    adapter._raw_rgb_frames.reset(frame)

    observation = adapter.observe()

    assert observation.data is frame
    assert observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY].shape == (2, 4, 4, 3)
    assert np.all(observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] == 7)


def test_deadly_corridor_requires_known_runtime_buttons():
    with pytest.raises(ValueError, match="available buttons"):
        _deadly_runtime_button_names(object())
    with pytest.raises(KeyError, match="JUMP"):
        _semantic_to_runtime_multibinary([0] * 7, ["JUMP"])
