"""Causal action history sampled at completed decision boundaries."""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Discrete, MultiBinary, Tuple


class DecisionActionHistory:
    """Encode admission, admitted command, and last applied action for each decision."""

    def __init__(self, action_space, *, num_envs: int, decisions: int):
        self._multibinary = isinstance(action_space, MultiBinary)
        if isinstance(action_space, Discrete):
            self.action_sizes = (action_space.n,)
        elif isinstance(action_space, Tuple) and all(isinstance(space, Discrete) for space in action_space.spaces):
            self.action_sizes = tuple(space.n for space in action_space.spaces)
        elif self._multibinary and action_space.shape == (7,):
            self.action_sizes = (3, 3, 3, 2)
        else:
            raise NotImplementedError(f"Decision action history does not support {action_space!r}")
        self.decisions = decisions
        self.action_dim = sum(size - 1 for size in self.action_sizes)
        self.step_dim = 1 + 2 * self.action_dim
        self.data = np.zeros((num_envs, decisions, self.step_dim), dtype=np.float32)
        self._basis = tuple(np.eye(size, dtype=np.float32)[:, 1:] for size in self.action_sizes)

    @property
    def observation_dim(self) -> int:
        return self.decisions * self.step_dim

    def reset(self, indices=None) -> None:
        if indices is None:
            self.data.fill(0)
        else:
            self.data[indices] = 0

    def append(self, indices, admitted, issued_actions, applied_actions) -> None:
        admitted = np.asarray(admitted, dtype=np.float32).reshape(-1)
        issued = self._encode(issued_actions) * admitted[:, None]
        applied = self._encode(applied_actions)
        rows = self.data[indices].copy()
        rows[:, :-1] = rows[:, 1:]
        rows[:, -1, 0] = admitted
        rows[:, -1, 1 : 1 + self.action_dim] = issued
        rows[:, -1, 1 + self.action_dim :] = applied
        self.data[indices] = rows

    def observation(self) -> np.ndarray:
        return self.data.reshape(self.data.shape[0], self.observation_dim).copy()

    def _encode(self, actions) -> np.ndarray:
        if self._multibinary:
            # The VLA button order is move, strafe, turn, attack; teacher history
            # encodes turn, move, strafe, attack. Keep both opposing bits if issued.
            return np.asarray(actions, dtype=np.float32).reshape(-1, 7)[:, [4, 5, 0, 1, 2, 3, 6]]
        values = np.asarray(actions, dtype=np.int64).reshape(-1, len(self.action_sizes))
        return np.concatenate(
            [basis[values[:, index]] for index, basis in enumerate(self._basis)], axis=1
        )
