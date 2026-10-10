"""User-owned module copied outside the checkout to exercise real worker imports."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box, Discrete

from latency_bench.core.types import Action, PolicyOutput
from latency_bench.envs.adapters.dummy import DummyEnvAdapter
from latency_bench.policy.base import PolicyRunner


assert not {"training", "sample_factory", "starVLA"}.intersection(sys.modules)


def _record(config, event):
    path = Path(config["factory_audit_path"])
    with path.open("a") as stream:
        stream.write(json.dumps({"event": event, "pid": os.getpid()}) + "\n")


class UserEnv(DummyEnvAdapter):
    """Record construction and cleanup in the process that owns this environment."""

    action_space = Discrete(512)

    def __init__(self, config, noop_action):
        super().__init__(env_fps=config["env"]["env_fps"], max_steps=4, noop_action=noop_action)
        self.config = config
        _record(config, "env_created")

    def close(self):
        super().close()
        _record(self.config, "env_closed")


class UserPolicy(PolicyRunner):
    """Count decisions within each episode to expose reset-state behavior."""

    def __init__(self, config):
        self.count = 0
        _record(config, "policy_created")

    def reset_state(self, slot_id=None):
        self.count = 0

    def predict(self, observation):
        self.count += 1
        return PolicyOutput(Action(self.count), raw_output=self.count)


def make_env(config, noop_action):
    return UserEnv(config, noop_action)


def make_policy(config, action_resolver):
    return UserPolicy(config)


def fail_policy(config, action_resolver):
    raise RuntimeError("user policy construction failed")


class CollectionEnv(UserEnv):
    """Reuse observation arrays to expose missing copies in rollout recording."""

    def __init__(self, config, noop_action):
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        super().__init__(config, noop_action)
        if "action_space" in config["env"]:
            self.action_space = Box(-5.0, 5.0, (1,), dtype=np.float32)

    def _make_observation(self, metadata=None):
        observation = super()._make_observation(metadata)
        self.frame.fill(self.env_step)
        observation.data["rgb"] = self.frame
        observation.data["student_state"] = np.asarray([self.env_step], dtype=np.float32)
        return observation


class CollectionPolicy(UserPolicy):
    """Predict from observed state for serial and concurrent collection."""

    def __init__(self, config):
        super().__init__(config)
        self.continuous = "action_space" in config["env"]

    def predict(self, observation):
        command = observation.data["state"] + 1
        return PolicyOutput(Action(np.asarray([command], dtype=np.float32) if self.continuous else command))

    def predict_batch(self, observations):
        return [self.predict(observation) for observation in observations]


def make_collection_env(config, noop_action):
    return CollectionEnv(config, noop_action)


def make_collection_policy(config, action_resolver):
    return CollectionPolicy(config)
