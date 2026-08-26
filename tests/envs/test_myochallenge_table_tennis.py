from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np

import latency_bench.envs.myochallenge_table_tennis as myo_env
from latency_bench.envs.myosuite import (
    MYOSUITE_ENV_ID,
    MyoSuiteEnv,
    make_myo_suite_env,
)
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.envs.gymnasium_task import gymnasium_task_contract
from scripts.scaffold_myochallenge import build_myochallenge_env_config


class _FakeMyoEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    observation_space = gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)

    def __init__(self) -> None:
        self.sim = _FakeSim()
        self.spec = SimpleNamespace(max_episode_steps=200)
        self.actions: list[np.ndarray] = []
        self.seed_calls: list[int] = []

    def seed(self, seed: int) -> list[int]:
        self.seed_calls.append(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {"seed": seed}

    def step(self, action: np.ndarray):
        self.actions.append(action)
        return np.ones(2, dtype=np.float32), 2.0, False, False, {"solved": np.bool_(True)}

    def close(self) -> None:
        return None


class _FakeSim:
    def __init__(self) -> None:
        self.model = SimpleNamespace(ptr=object())
        self.data = SimpleNamespace(ptr=object())
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeRenderer:
    instances: list[_FakeRenderer] = []

    def __init__(self, model: object, *, height: int, width: int) -> None:
        self.model = model
        self.height = height
        self.width = width
        self.scenes: list[tuple[object, str]] = []
        self.close_calls = 0
        self.__class__.instances.append(self)

    def update_scene(self, data: object, *, camera: str) -> None:
        self.scenes.append((data, camera))

    def render(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), 7, dtype=np.uint8)

    def close(self) -> None:
        self.close_calls += 1


def test_wrapper_converts_box_actions_records_score_and_renders(monkeypatch):
    native_env = _FakeMyoEnv()
    monkeypatch.setattr(myo_env.gym, "make", lambda *args, **kwargs: native_env)
    monkeypatch.setitem(__import__("sys").modules, "mujoco", SimpleNamespace(Renderer=_FakeRenderer))

    env = myo_env.MyoChallengeTableTennisP2Env(
        base_env_id="myoChallengeTableTennisP2-v0",
        render_mode="rgb_array",
        base_make_kwargs={},
    )
    env.reset(seed=4)
    observation, reward, terminated, truncated, info = env.step([2.0, -0.5, 0.25])
    frame = env.render()

    expected_action = np.asarray([2.0, -0.5, 0.25], dtype=np.float32)
    assert np.array_equal(native_env.actions[0], expected_action)
    assert native_env.seed_calls == [4]
    assert info["solved"] is True
    assert info["score"] == 1.0
    assert observation.shape == (2,)
    assert reward == 2.0
    assert terminated is False
    assert truncated is False
    assert frame.shape == (224, 224, 3)
    assert frame.dtype == np.uint8
    renderer = _FakeRenderer.instances[-1]
    assert renderer.scenes[-1][1] == "default"

    env.close()
    assert renderer.close_calls == 1
    assert native_env.sim.close_calls == 1


def test_shared_wrapper_preserves_native_info_without_task_metrics(monkeypatch):
    native_env = _FakeMyoEnv()
    monkeypatch.setattr(myo_env.gym, "make", lambda *args, **kwargs: native_env)

    env = MyoSuiteEnv(
        base_env_id="myoChallengeBaodingP1-v1",
        render_mode="rgb_array",
        base_make_kwargs={},
        camera_name="default",
        render_width=32,
        render_height=32,
    )
    env.reset(seed=4)
    _, _, _, _, info = env.step([0.0, 0.0, 0.0])

    assert native_env.seed_calls == [4]
    assert np.array_equal(native_env.actions[0], np.zeros(3, dtype=np.float32))
    assert info == {"solved": np.bool_(True)}
    assert "score" not in info
    env.close()


def test_generic_myo_registration_and_box_contract():
    spec = gym.spec(MYOSUITE_ENV_ID)
    assert spec.entry_point is make_myo_suite_env

    report = {
        "action_space": {
            "labels": ["muscle_0", "muscle_1"],
            "low": [0.0, 0.0],
            "high": [1.0, 1.0],
            "dtype": "float32",
        },
        "noop_action": [0.0, 0.0],
        "timing": {"native_env_fps": 40.0},
    }
    contract = build_myochallenge_env_config(
        report=report,
        task_name="myochallenge_baoding_p1",
        benchmark_env_id=MYOSUITE_ENV_ID,
        base_env_id="myoChallengeBaodingP1-v1",
        registration_imports=["latency_bench.envs.myosuite"],
        base_make_kwargs={"normalize_act": True},
        camera_name="default",
        render_width=224,
        render_height=224,
        base_prompt="Rotate both balls without dropping them.",
        env_fps=40.0,
        obs_fps=20.0,
        frame_stack=1,
    )

    assert contract["action_space"] == {
        "type": "box",
        "labels": ["muscle_0", "muscle_1"],
        "low": [0.0, 0.0],
        "high": [1.0, 1.0],
        "dtype": "float32",
    }
    assert contract["noop_action"] == [0.0, 0.0]
    assert contract["make_kwargs"]["base_env_id"] == "myoChallengeBaodingP1-v1"
    assert contract["env_fps"] == 40.0
    assert contract["obs_fps"] == 20.0
    task_contract = gymnasium_task_contract(contract)
    assert task_contract["action_space"]["low"] == [0.0, 0.0]
    assert task_contract["noop_action"] == [0.0, 0.0]


def test_registration_is_lazy_and_exposes_benchmark_id():
    spec = gym.spec(myo_env.GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID)
    assert spec.max_episode_steps is None


def test_myochallenge_configs_keep_native_100_hz_contract():
    config_dir = Path("configs/examples/gymnasium/myochallenge_table_tennis_p2")
    config_paths = sorted(config_dir.glob("*.yaml"))
    assert config_paths
    for config_path in config_paths:
        config = load_config(config_path)
        env_cfg = config["env"]
        assert env_cfg["env_fps"] == 100.0
        assert env_cfg["obs_fps"] == 100.0
        assert env_cfg["env_id"] == myo_env.GYMNASIUM_MYOCHALLENGE_TABLE_TENNIS_P2_ENV_ID
        assert env_cfg["action_space"]["type"] == "box"
        assert len(env_cfg["action_space"]["labels"]) == 275
        assert env_cfg["action_space"]["labels"][-2:] == ["pelvis_x", "pelvis_y"]
        assert len(env_cfg["noop_action"]) == 275
        assert EnvClock(env_fps=env_cfg["env_fps"], obs_fps=env_cfg["obs_fps"]).obs_stride_raw_frames == 1


def test_fixed_50ms_training_config_uses_exact_raw_frame_delay():
    config = load_config(
        "configs/examples/gymnasium/myochallenge_table_tennis_p2/small_model_train_fixed_50ms.yaml"
    )
    assert config["latency"] == {
        "method": "fixed",
        "fixed_latency_ms": 50.0,
        "add_latency_info": False,
    }


def test_myochallenge_small_model_evals_cover_requested_raw_frame_delays():
    expected_experiments = {
        "small_model_eval.yaml": "gymnasium_myochallenge_table_tennis_p2_zero_latency",
        "small_model_eval_fixed_50ms.yaml": "gymnasium_myochallenge_table_tennis_p2_fixed_50ms",
    }
    for name, expected_experiment in expected_experiments.items():
        config = load_config(
            Path("configs/examples/gymnasium/myochallenge_table_tennis_p2") / name
        )
        assert config["experiment"]["name"] == expected_experiment
        assert config["policy"]["type"] == "gymnasium_sf"
        assert config["evaluation"]["eval_latency_values"] == [0, 1, 2, 5, 10]
