from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Discrete, MultiDiscrete, Tuple as SpaceTuple

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action


def _is_flat_discrete_tuple_space(action_space: Any) -> bool:
    return isinstance(action_space, SpaceTuple) and all(
        isinstance(subspace, Discrete)
        for subspace in action_space.spaces
    )


def _is_one_dimensional_multidiscrete_space(action_space: Any) -> bool:
    return isinstance(action_space, MultiDiscrete) and np.asarray(action_space.nvec).ndim == 1


def is_supported_batched_policy_action_space(action_space: Any) -> bool:
    return (
        isinstance(action_space, Discrete)
        or _is_flat_discrete_tuple_space(action_space)
        or _is_one_dimensional_multidiscrete_space(action_space)
    )


def deterministic_actions_for_space(
    action_distribution: Any,
    *,
    action_space: Any,
    argmax_actions,
    device: torch.device,
) -> torch.Tensor:
    if _is_flat_discrete_tuple_space(action_space):
        component_actions = []
        for component_distribution in action_distribution.distributions:
            component = _as_action_tensor(argmax_actions(component_distribution), device)
            if component.ndim == 0:
                component = component.reshape(1)
            if component.ndim == 2 and component.shape[-1] == 1:
                component = component.squeeze(-1)
            component_actions.append(component)
        return torch.stack(component_actions, dim=1)
    return _as_action_tensor(argmax_actions(action_distribution), device)


def _as_action_tensor(actions: Any, device: torch.device) -> torch.Tensor:
    if isinstance(actions, torch.Tensor):
        return actions.to(device=device)
    return torch.as_tensor(actions, device=device)


def normalize_actions_for_space(
    actions: Any,
    *,
    batch_size: int,
    action_space: Any,
    device: torch.device,
    unsqueeze_tensor,
) -> torch.Tensor:
    actions = _as_action_tensor(actions, device)
    if isinstance(action_space, Discrete):
        if actions.ndim == 0:
            if batch_size != 1:
                raise ValueError(
                    "Sample Factory batched policy returned a scalar action for multiple observations"
                )
            actions = actions.reshape(1, 1)
        elif actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)
        _require_action_shape(actions, batch_size=batch_size, num_components=1)
        return actions

    if _is_flat_discrete_tuple_space(action_space):
        actions = _normalize_composite_action_tensor(
            actions,
            batch_size=batch_size,
            num_components=len(action_space.spaces),
            action_space_name="Tuple",
        )
        _validate_discrete_tuple_action_values(actions, action_space)
        return actions

    if _is_one_dimensional_multidiscrete_space(action_space):
        actions = _normalize_composite_action_tensor(
            actions,
            batch_size=batch_size,
            num_components=len(action_space.nvec),
            action_space_name="MultiDiscrete",
        )
        _validate_multidiscrete_action_values(actions, action_space)
        return actions

    raise NotImplementedError(f"Unsupported Sample Factory action space for batched prediction: {action_space!r}")


def _normalize_composite_action_tensor(
    actions: torch.Tensor,
    *,
    batch_size: int,
    num_components: int,
    action_space_name: str,
) -> torch.Tensor:
    if actions.ndim == 1:
        if batch_size == 1 and actions.shape[0] == num_components:
            actions = actions.reshape(1, num_components)
        else:
            raise ValueError(
                f"Sample Factory {action_space_name} actions must have shape [batch, {num_components}], "
                f"got {tuple(actions.shape)}"
            )
    if actions.ndim != 2:
        raise ValueError(
            f"Sample Factory {action_space_name} actions must have shape [batch, {num_components}], "
            f"got {tuple(actions.shape)}"
        )
    _require_action_shape(actions, batch_size=batch_size, num_components=num_components)
    return actions


def _require_action_shape(actions: torch.Tensor, *, batch_size: int, num_components: int) -> None:
    expected_shape = (batch_size, num_components)
    if tuple(actions.shape) != expected_shape:
        raise ValueError(f"Sample Factory actions must have shape {expected_shape}, got {tuple(actions.shape)}")


def _validate_discrete_tuple_action_values(actions: torch.Tensor, action_space: SpaceTuple) -> None:
    for component_index, subspace in enumerate(action_space.spaces):
        values = actions[:, component_index]
        if torch.any(values < 0) or torch.any(values >= subspace.n):
            raise ValueError(
                f"Tuple action component {component_index} is outside Discrete({subspace.n})"
            )


def _validate_multidiscrete_action_values(actions: torch.Tensor, action_space: MultiDiscrete) -> None:
    nvec = np.asarray(action_space.nvec).tolist()
    for component_index, n in enumerate(nvec):
        values = actions[:, component_index]
        if torch.any(values < 0) or torch.any(values >= n):
            raise ValueError(
                f"MultiDiscrete action component {component_index} is outside range [0, {n})"
            )


def split_batch_actions(processed_actions: Any, batch_size: int, *, action_space: Any) -> list[Any]:
    if isinstance(processed_actions, torch.Tensor):
        processed_actions = processed_actions.detach().cpu().numpy()
    if isinstance(processed_actions, np.ndarray):
        if processed_actions.ndim == 0:
            if batch_size != 1:
                raise ValueError(
                    "Sample Factory batched policy returned a scalar processed action for multiple observations"
                )
            return [to_python_action(processed_actions)]
        if processed_actions.shape[0] != batch_size:
            raise ValueError(
                "Sample Factory batched policy returned the wrong number of processed actions: "
                f"expected {batch_size}, got {processed_actions.shape[0]}"
            )
        return [to_python_action(processed_actions[index]) for index in range(batch_size)]
    if isinstance(processed_actions, list):
        if _is_flat_discrete_tuple_space(action_space) and len(processed_actions) == len(action_space.spaces):
            return _split_tuple_component_batches(processed_actions, batch_size)
        if len(processed_actions) != batch_size:
            raise ValueError(
                "Sample Factory batched policy returned the wrong number of processed actions: "
                f"expected {batch_size}, got {len(processed_actions)}"
            )
        return [to_python_action(item) for item in processed_actions]
    if batch_size == 1:
        return [to_python_action(processed_actions)]
    raise ValueError(
        "Sample Factory batched policy returned an unsupported processed action batch "
        f"of type {type(processed_actions).__name__}"
    )


def _split_tuple_component_batches(component_batches: list[Any], batch_size: int) -> list[Any]:
    components = []
    for component in component_batches:
        if isinstance(component, torch.Tensor):
            component = component.detach().cpu().numpy()
        component = np.asarray(component)
        if component.ndim == 0:
            if batch_size != 1:
                raise ValueError(
                    "Sample Factory Tuple component returned scalar action for multiple observations"
                )
            component = component.reshape(1)
        if component.shape[0] != batch_size:
            raise ValueError(
                "Sample Factory Tuple component returned the wrong batch size: "
                f"expected {batch_size}, got {component.shape[0]}"
            )
        components.append([to_python_action(component[index]) for index in range(batch_size)])
    return [
        [to_python_action(component_values[slot_index]) for component_values in components]
        for slot_index in range(batch_size)
    ]


def to_python_action(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0].item()
        squeezed = np.squeeze(value, axis=0) if value.ndim > 0 and value.shape[0] == 1 else value
        if squeezed.ndim == 0:
            return squeezed.item()
        return squeezed.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return to_python_action(value[0])
        return type(value)(to_python_action(item) for item in value)
    return value


def resolve_action(raw_value: Any, action_resolver: ActionResolver) -> Action:
    if isinstance(raw_value, list):
        raw_value = tuple(raw_value)
    return action_resolver.resolve(raw_value)
