from __future__ import annotations

import builtins
import json
import runpy
import sys
from pathlib import Path

import numpy as np

from scripts.mikasa import evaluate


def test_spawn_bootstrap_does_not_import_mikasa_dependencies(monkeypatch):
    real_import = builtins.__import__

    def reject_mikasa_imports(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"baselines", "mani_skill", "mikasa_robo_suite"}:
            raise AssertionError(f"spawn bootstrap imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_mikasa_imports)
    runpy.run_path(evaluate.__file__, run_name="__mp_main__")


class _FakeEnv:
    def __init__(self):
        self.reset_seeds = []
        self.episode_steps = (
            ((False, 1.0, False), (True, 2.0, True)),
            ((False, 3.0, True),),
        )
        self.episode_index = -1
        self.step_index = 0
        self.closed = False

    def reset(self, *, seed):
        self.reset_seeds.append(seed)
        self.episode_index += 1
        self.step_index = 0
        return {"state": np.zeros((1, 1), dtype=np.float32)}, {}

    def step(self, action):
        success, reward, terminated = self.episode_steps[self.episode_index][self.step_index]
        self.step_index += 1
        return (
            {"state": np.zeros((1, 1), dtype=np.float32)},
            np.asarray([reward], dtype=np.float32),
            np.asarray([terminated]),
            np.asarray([False]),
            {"success": np.asarray([success])},
        )

    def close(self):
        self.closed = True


class _FakePolicy:
    def __init__(self):
        self.observations = []

    def forward(self, observation):
        self.observations.append(observation)
        return np.zeros((1, 7), dtype=np.float32)


def test_zero_delay_evaluation_writes_success_once_summary(tmp_path: Path, monkeypatch):
    env = _FakeEnv()
    policy = _FakePolicy()
    monkeypatch.setattr(evaluate, "_make_ppo_env", lambda: env)
    monkeypatch.setattr(evaluate, "_PpoPolicy", lambda env, checkpoint: policy)

    output_dir = tmp_path / "nested" / "eval"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--policy",
            "ppo",
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--episodes",
            "2",
            "--output-dir",
            str(output_dir),
        ],
    )
    evaluate.main()
    summary = json.loads((output_dir / "summary.json").read_text())

    assert env.reset_seeds == [evaluate.START_SEED, evaluate.START_SEED + 1]
    assert len(policy.observations) == 3
    assert env.closed
    assert summary == {
        "seeds": [evaluate.START_SEED, evaluate.START_SEED + 1],
        "successes": [True, False],
        "success_rate": 0.5,
        "returns": [3.0, 3.0],
        "lengths": [2, 1],
    }


def test_latency_backend_exposes_two_views_and_direct_action():
    class FakeProfileEnv:
        def __init__(self):
            self.actions = []
            self.closed = False
            self.step_index = 0
            self.unwrapped = self
            self.device = "cpu"

        def reset(self, *, seed, options=None):
            return {"rgb": np.zeros((1, 2, 3, 6), dtype=np.uint8)}, {}

        def step(self, action):
            self.actions.append(action)
            rgb = np.zeros((1, 2, 3, 6), dtype=np.uint8)
            rgb[..., :3] = 11
            rgb[..., 3:] = 22
            success = self.step_index == 0
            self.step_index += 1
            return rgb_to_observation(rgb), np.asarray([1.5]), np.asarray([False]), np.asarray([False]), {
                "success": np.asarray([success])
            }

        def close(self):
            self.closed = True

    def rgb_to_observation(rgb):
        return {"rgb": rgb}

    raw_env = FakeProfileEnv()
    env = evaluate.MikasaEnvStepBackend(
        config={
            "env": {
                "env_fps": 20,
                "noop_action": [0.0] * 7,
            }
        },
        num_slots=1,
        env=raw_env,
    )
    reset_observation = env.reset_slot(0, episode_id=0, seed=123)
    first_result = env.step_slots({0: evaluate.Action(value=[0.1] * 7)})[0].result
    result = env.step_slots({0: evaluate.Action(value=[0.1] * 7)})[0].result

    assert reset_observation.metadata[evaluate.MIKASA_IMAGE_VIEWS_INFO_KEY].shape == (2, 2, 3, 3)
    assert [action.shape for action in raw_env.actions] == [(1, 7), (1, 7)]
    np.testing.assert_allclose(raw_env.actions[0], [[0.1] * 7])
    np.testing.assert_allclose(result.info["applied_action"], [0.1] * 7)
    observation = env.observe_slots([0])[0]
    assert [view.mean() for view in observation.metadata[evaluate.MIKASA_IMAGE_VIEWS_INFO_KEY]] == [
        11,
        22,
    ]
    assert result.reward == 1.5
    assert first_result.info["success"] is True
    assert result.info["success"] is False
    assert result.info["task_metrics"] == {"success": 1.0}
    assert result.info["applied_action_name"] is None
