from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence

from latency_bench.core.types import Action, Observation, PolicyOutput


class PolicyRunner(ABC):
    @abstractmethod
    def predict(self, observation: Observation) -> PolicyOutput:
        ...


class RandomPolicyRunner(PolicyRunner):
    def __init__(self, actions: Sequence[Action], seed: int | None = None):
        if not actions:
            raise ValueError("actions cannot be empty")
        self.actions = list(actions)
        self.rng = random.Random(seed)

    def predict(self, observation: Observation) -> PolicyOutput:
        action = self.rng.choice(self.actions)
        return PolicyOutput(action=action, raw_output=action.value)


class ConstantPolicyRunner(PolicyRunner):
    def __init__(self, action: Action):
        self.action = action

    def predict(self, observation: Observation) -> PolicyOutput:
        return PolicyOutput(action=self.action, raw_output=self.action.value)

