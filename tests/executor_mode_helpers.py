from __future__ import annotations

import time

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, Observation, OrderingPolicy, PolicyOutput
from latency_bench.envs.base import DummyEnvAdapter
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.persistence import HoldPolicy


class SequencePolicyRunner(PolicyRunner):
    def __init__(self, actions: list[Action], sleep_s: float = 0.0):
        if not actions:
            raise ValueError("actions cannot be empty")
        if sleep_s < 0:
            raise ValueError("sleep_s must be non-negative")
        self.actions = list(actions)
        self.sleep_s = float(sleep_s)
        self.index = 0

    def predict(self, observation: Observation) -> PolicyOutput:
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return PolicyOutput(action=action, raw_output=action.value)


def make_dummy_parts(max_steps: int = 5, env_fps: float = 30.0):
    noop_action = Action(value="noop", name="noop", is_noop=True)
    jump_action = Action(value="jump", name="jump")
    env = DummyEnvAdapter(env_fps=env_fps, max_steps=max_steps, noop_action=noop_action)
    clock = EnvClock(env_fps=env_fps, obs_fps=env_fps)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop_action,
        persistence_policy=HoldPolicy(),
        ordering_policy=OrderingPolicy.LATEST_READY,
    )
    return env, clock, noop_action, jump_action, action_scheduler
