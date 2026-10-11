from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence

from latency_bench.core.types import Action, Observation, PolicyOutput


class PolicyRunner(ABC):
    """Predict actions from benchmark observations and own per-episode policy state."""

    @abstractmethod
    def predict(self, observation: Observation) -> PolicyOutput:
        ...

    def reset_state(self, slot_id: int | None = None) -> None:
        return None

    def predict_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement safe batched prediction. "
            "Implement predict_batch() for stateful policies instead of looping predict()."
        )


class RandomPolicyRunner(PolicyRunner):
    def __init__(self, actions: Sequence[Action], seed: int | None = None):
        self.actions = list(actions)
        self.rng = random.Random(seed)

    def predict(self, observation: Observation) -> PolicyOutput:
        action = self.rng.choice(self.actions)
        return PolicyOutput(action=action, raw_output=action.value)

    def predict_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        return [self.predict(observation) for observation in observations]


class ConstantPolicyRunner(PolicyRunner):
    def __init__(self, action: Action):
        self.action = action

    def predict(self, observation: Observation) -> PolicyOutput:
        return PolicyOutput(action=self.action, raw_output=self.action.value)

    def predict_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        return [
            PolicyOutput(action=self.action, raw_output=self.action.value)
            for _observation in observations
        ]
