"""Rollout drivers that adapt Sample Factory env outputs to teacher export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from latency_bench.integrations.sample_factory.actions import to_python_action
from latency_bench.utils.array import to_numpy


def batch_observations(
    observations: Sequence[Any],
    *,
    preserve_leading_batch: bool = False,
) -> Any:
    """Batch active environment observations for policy inference.

    Args:
        observations: Observations from active rollout slots.
        preserve_leading_batch: Keep Sample Factory's leading singleton batch
            axis for generic vector observations.

    Returns:
        Batched observation object matching Sample Factory expectations.
    """
    first = observations[0]
    if isinstance(first, Mapping):
        if len(observations) == 1:
            arrays = {key: to_numpy(value) for key, value in first.items()}
            if any(array.ndim >= 4 for array in arrays.values()) or (
                preserve_leading_batch
                and all(
                    array.ndim >= 1 and int(array.shape[0]) == 1
                    for array in arrays.values()
                )
            ):
                return arrays
        arrays = {}
        for key in first:
            values = [to_numpy(obs[key]) for obs in observations]
            arrays[key] = (
                np.concatenate(values, axis=0)
                if (
                    values[0].ndim >= 4
                    or (
                        preserve_leading_batch
                        and values[0].ndim >= 1
                        and int(values[0].shape[0]) == 1
                    )
                )
                else np.stack(values, axis=0)
            )
        return arrays
    return {"obs": np.stack([to_numpy(obs) for obs in observations], axis=0)}


def _to_host_obs(observation: Any) -> Any:
    """Move a batched observation to host NumPy once, preserving structure.

    Converting the whole batch a single time (instead of slicing the device
    tensor per slot) avoids thousands of tiny GPU->CPU syncs per step.

    Args:
        observation: Batched observation object (array, tensor, or mapping).

    Returns:
        Host NumPy observation with the same structure.
    """
    if isinstance(observation, Mapping):
        return {key: _to_host_obs(value) for key, value in observation.items()}
    return to_numpy(observation)


def _obs_slice(host_obs: Any, index: int) -> Any:
    """Select one agent observation from a host-resident batched observation.

    Args:
        host_obs: Host NumPy batched observation (see ``_to_host_obs``).
        index: Agent slot index.

    Returns:
        Single-agent observation preserving the leading batch dimension.
    """
    if isinstance(host_obs, Mapping):
        return {key: _obs_slice(value, index) for key, value in host_obs.items()}
    return host_obs[index : index + 1]


def _to_host_array(value: Any) -> Any:
    """Move a full batch of scalar results (reward/terminated/...) to host once.

    Args:
        value: Batched tensor, array, or sequence.

    Returns:
        Host NumPy array indexable per slot.
    """
    return to_numpy(value)


@dataclass
class RolloutStepResult:
    """Normalize one environment step result for rollout drivers.

    Args:
        next_obs: Observation after applying the action.
        reward: Step reward.
        terminated: Environment termination flag.
        truncated: Environment truncation flag.
        info: Step info mapping.
    """
    next_obs: Any
    reward: Any
    terminated: Any
    truncated: Any
    info: Mapping[str, Any]


class SerialTeacherRolloutDriver:
    """Rollout driver for multiple independent environment instances."""

    def __init__(self, envs: Sequence[Any]):
        """Create a driver that steps independent env instances.

        Args:
            envs: Sequence of Sample Factory-compatible environments.
        """
        self.envs = list(envs)

    @property
    def slot_count(self) -> int:
        """Return the number of independently stepped rollout slots."""
        return len(self.envs)

    @property
    def action_space(self) -> Any:
        """Return the action space shared by the serial envs."""
        return self.envs[0].action_space

    def reset_slots(
        self,
        reset_plan: Mapping[int, int],
    ) -> dict[int, tuple[Any, Mapping[str, Any]]]:
        """Reset selected serial rollout slots.

        Args:
            reset_plan: Mapping from slot id to reset seed.

        Returns:
            Mapping from slot id to observation and reset info.
        """

        observations = {}
        for slot, seed in reset_plan.items():
            obs, reset_info = self.envs[int(slot)].reset(seed=int(seed))
            observations[int(slot)] = (obs, reset_info[0])
        return observations

    def step(self, active_slots: Sequence[int], raw_actions: Sequence[Any]) -> dict[int, RolloutStepResult]:
        """Step selected serial rollout slots.

        Args:
            active_slots: Slot ids to step.
            raw_actions: Actions aligned with ``active_slots``.

        Returns:
            Step results keyed by slot id.
        """

        step_results = {}
        for batch_idx, slot in enumerate(active_slots):
            next_obs, reward, terminated, truncated, info = self.envs[int(slot)].step([raw_actions[batch_idx]])
            step_results[int(slot)] = RolloutStepResult(
                next_obs=next_obs,
                reward=to_python_action(reward),
                terminated=to_python_action(terminated),
                truncated=to_python_action(truncated),
                info=info[0],
            )
        return step_results

    def continue_after_life_loss(self, slot: int) -> tuple[Any, Mapping[str, Any]]:
        """Advance a serial Atari env from a life-loss terminal into the next life."""
        obs, reset_info = self.envs[int(slot)].reset()
        return obs, reset_info[0]

    def close(self) -> None:
        """Close all serial environments."""
        for env in self.envs:
            env.close()


class BatchedTeacherRolloutDriver:
    """Rollout driver for one batched Sample Factory environment."""

    def __init__(self, env: Any, slot_count: int):
        """Create a driver for one batched environment.

        Args:
            env: Batched environment instance.
            slot_count: Number of active agents managed by the env.
        """
        self.env = env
        self.control_env = env.unwrapped
        self._slot_count = int(slot_count)
        self._reset_started = False
        self._last_step_results: dict[int, RolloutStepResult] = {}

    @property
    def slot_count(self) -> int:
        """Return the number of batched rollout slots."""
        return self._slot_count

    @property
    def action_space(self) -> Any:
        """Return the batched environment action space."""
        return self.env.action_space

    def reset_slots(
        self,
        reset_plan: Mapping[int, int],
        active_mask: Sequence[bool],
    ) -> dict[int, tuple[Any, Mapping[str, Any]]]:
        """Reset selected agents in the batched env.

        Args:
            reset_plan: Mapping from slot id to reset seed.
            active_mask: Active agent mask after resets are applied.

        Returns:
            Mapping from slot id to observation and reset info.
        """
        if not self._reset_started:
            first_seed = reset_plan[0]
            obs_batch, info_batch = self.env.reset(seed=first_seed)
            self._reset_started = True
            non_sequential_plan = {
                slot: seed
                for slot, seed in reset_plan.items()
                if seed != first_seed + slot
            }
            if non_sequential_plan:
                obs_batch, info_batch = self.control_env.reset_agents(
                    non_sequential_plan,
                    active_mask=list(active_mask),
                )
            elif list(active_mask) != [True for _ in range(self._slot_count)]:
                self.control_env.set_active_agents(list(active_mask))
            obs_host = _to_host_obs(obs_batch)
            return {
                slot: (
                    _obs_slice(obs_host, slot),
                    dict(info_batch[slot]),
                )
                for slot in reset_plan
            }

        obs_batch, info_batch = self.control_env.reset_agents(
            reset_plan,
            active_mask=list(active_mask),
        )
        obs_host = _to_host_obs(obs_batch)
        return {
            slot: (
                _obs_slice(obs_host, slot),
                dict(info_batch[slot]),
            )
            for slot in reset_plan
        }

    def step(self, active_slots: Sequence[int], raw_actions: Sequence[Any]) -> dict[int, RolloutStepResult]:
        """Step active agents in the batched env.

        Args:
            active_slots: Slot ids to step.
            raw_actions: Actions aligned with ``active_slots``.

        Returns:
            Step results keyed by slot id.
        """
        full_actions = [0 for _ in range(self._slot_count)]
        for batch_idx, slot in enumerate(active_slots):
            full_actions[slot] = raw_actions[batch_idx]
        next_obs_batch, rewards, terminated, truncated, infos = self.env.step(full_actions)
        # Move each batched result to host once, then index per slot (cheap host
        # views). Slicing the device tensors per slot would re-copy the whole
        # batch on every call -> thousands of GPU->CPU syncs per step.
        obs_host = _to_host_obs(next_obs_batch)
        rewards_host = _to_host_array(rewards)
        terminated_host = _to_host_array(terminated)
        truncated_host = _to_host_array(truncated)
        results = {
            slot: RolloutStepResult(
                next_obs=_obs_slice(obs_host, slot),
                reward=to_python_action(rewards_host[slot]),
                terminated=to_python_action(terminated_host[slot]),
                truncated=to_python_action(truncated_host[slot]),
                info=infos[slot],
            )
            for slot in active_slots
        }
        self._last_step_results = results
        return results

    def continue_after_life_loss(self, slot: int) -> tuple[Any, Mapping[str, Any]]:
        """Return the post-soft-reset observation already produced by the batched env."""

        step_result = self._last_step_results[slot]
        return step_result.next_obs, step_result.info

    def sync_active(self, active_mask: Sequence[bool]) -> None:
        """Update the batched env's active agent mask.

        Args:
            active_mask: Active agent mask.
        """
        self.control_env.set_active_agents(list(active_mask))

    def close(self) -> None:
        """Close the batched environment."""
        self.env.close()
