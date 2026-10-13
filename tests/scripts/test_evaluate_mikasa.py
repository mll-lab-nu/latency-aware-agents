from __future__ import annotations

import builtins
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from latency_bench.envs.adapters import mikasa as mikasa_adapter
from latency_bench.envs.tasks import mikasa as mikasa_task

from scripts.mikasa import evaluate


@pytest.mark.parametrize("module", [evaluate, mikasa_task, mikasa_adapter])
def test_spawn_bootstrap_does_not_import_mikasa_dependencies(monkeypatch, module):
    real_import = builtins.__import__

    def reject_mikasa_imports(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"baselines", "mani_skill", "mikasa_robo_suite"}:
            raise AssertionError(f"spawn bootstrap imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_mikasa_imports)
    runpy.run_path(module.__file__, run_name="__mp_main__")


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
    monkeypatch.setattr(evaluate, "make_ppo_env", lambda: env)
    monkeypatch.setattr(evaluate, "MikasaPpoPolicy", lambda env, checkpoint: policy)

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
