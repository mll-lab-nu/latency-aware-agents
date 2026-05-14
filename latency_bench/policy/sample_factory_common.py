from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from gymnasium.spaces import Box, Dict as SpaceDict, Discrete, MultiDiscrete, Tuple as SpaceTuple

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.policy.base import PolicyRunner
from training.common.checkpoints import load_sample_factory_checkpoint_from_paths, trusted_torch_load


def _resolve_device(device: Any) -> str:
    device_name = str(device).strip().lower()
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "gpu":
        return "cuda"
    return device_name


def resolve_sample_factory_device(device: Any) -> str:
    device_name = str(device).strip().lower()
    if device_name == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        return "gpu"
    return device_name


def _obs_to_sf_inputs(
    observation: Observation,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    data = observation.data
    action_mask = None
    if isinstance(data, Mapping):
        obs_dict: dict[str, torch.Tensor] = {}
        for key, value in data.items():
            if str(key) == "action_mask":
                action_mask = torch.as_tensor(value, device=device).unsqueeze(0)
                continue
            tensor = torch.as_tensor(value, device=device)
            if tensor.ndim >= 1:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.reshape(1)
            obs_dict[str(key)] = tensor
        return obs_dict, action_mask

    tensor = torch.as_tensor(data, device=device)
    if tensor.ndim >= 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.reshape(1)
    return {"obs": tensor}, action_mask


def _obs_batch_to_sf_inputs(
    observations: Sequence[Observation],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    if not observations:
        return {}, None

    stacked_inputs = _obs_batch_to_sf_inputs_stacked(observations, device)
    if stacked_inputs is not None:
        return stacked_inputs

    batched_parts: dict[str, list[torch.Tensor]] = {}
    expected_keys: set[str] | None = None
    action_masks: list[torch.Tensor | None] = []
    for index, observation in enumerate(observations):
        obs_dict, action_mask = _obs_to_sf_inputs(observation, device)
        keys = set(obs_dict)
        if expected_keys is None:
            expected_keys = keys
            for key in keys:
                batched_parts[key] = []
        elif keys != expected_keys:
            raise ValueError(
                "Cannot batch Sample Factory observations with different keys: "
                f"expected {sorted(expected_keys)}, got {sorted(keys)} at batch index {index}"
            )
        for key, tensor in obs_dict.items():
            batched_parts[key].append(tensor)
        action_masks.append(action_mask)

    try:
        batched_obs = {
            key: torch.cat(tensors, dim=0)
            for key, tensors in batched_parts.items()
        }
    except RuntimeError as exc:
        raise ValueError("Cannot batch Sample Factory observations with incompatible tensor shapes") from exc

    if all(mask is not None for mask in action_masks):
        return batched_obs, torch.cat([mask for mask in action_masks if mask is not None], dim=0)
    return batched_obs, None


def _obs_batch_to_sf_inputs_stacked(
    observations: Sequence[Observation],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None] | None:
    first_data = observations[0].data
    if isinstance(first_data, Mapping):
        return _mapping_obs_batch_to_sf_inputs_stacked(observations, device)
    return _plain_obs_batch_to_sf_inputs_stacked(observations, device)


def _plain_obs_batch_to_sf_inputs_stacked(
    observations: Sequence[Observation],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], None] | None:
    values = []
    for observation in observations:
        data = observation.data
        if isinstance(data, Mapping) or isinstance(data, torch.Tensor):
            return None
        values.append(data)
    stacked = _stack_numpy_values(values)
    if stacked is None:
        return None
    return {"obs": torch.as_tensor(stacked, device=device)}, None


def _mapping_obs_batch_to_sf_inputs_stacked(
    observations: Sequence[Observation],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None] | None:
    first_data = observations[0].data
    if not isinstance(first_data, Mapping):
        return None
    keys = [str(key) for key in first_data if str(key) != "action_mask"]
    obs_values: dict[str, list[Any]] = {key: [] for key in keys}
    action_mask_values = []
    all_have_action_mask = True

    for observation in observations:
        data = observation.data
        if not isinstance(data, Mapping):
            return None
        current_values = {
            str(key): value
            for key, value in data.items()
            if str(key) != "action_mask"
        }
        if list(current_values) != keys:
            return None
        for key in keys:
            value = current_values[key]
            if isinstance(value, torch.Tensor):
                return None
            obs_values[key].append(value)
        has_action_mask = False
        action_mask = None
        for key, value in data.items():
            if str(key) == "action_mask":
                has_action_mask = True
                action_mask = value
                break
        if not has_action_mask:
            all_have_action_mask = False
        elif isinstance(action_mask, torch.Tensor):
            return None
        else:
            action_mask_values.append(action_mask)

    batched_obs = {}
    for key in keys:
        stacked = _stack_numpy_values(obs_values[key])
        if stacked is None:
            return None
        batched_obs[key] = torch.as_tensor(stacked, device=device)

    action_mask = None
    if all_have_action_mask:
        stacked_mask = _stack_numpy_values(action_mask_values)
        if stacked_mask is None:
            return None
        action_mask = torch.as_tensor(stacked_mask, device=device)
    return batched_obs, action_mask


def _stack_numpy_values(values: Sequence[Any]) -> np.ndarray | None:
    arrays = [np.asarray(value) for value in values]
    first = arrays[0]
    if first.dtype == object:
        return None
    first_shape = first.shape
    first_dtype = first.dtype
    for array in arrays[1:]:
        if array.shape != first_shape or array.dtype != first_dtype or array.dtype == object:
            return None
    return np.stack(arrays, axis=0)


def _extract_action_mask(observation: Observation, device: torch.device):
    metadata = getattr(observation, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    action_mask = metadata.get("action_mask")
    if action_mask is None:
        return None
    return torch.as_tensor(action_mask, device=device).unsqueeze(0)


def _checkpoint_state_dict_from_path(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = trusted_torch_load(path, device)
    return checkpoint


def _load_sf_policy_state_dict(cfg, actor_critic, device: torch.device, checkpoint_path: str | None) -> dict[str, Any]:
    from sample_factory.algo.learning.learner import Learner

    if checkpoint_path:
        checkpoint_dict = _checkpoint_state_dict_from_path(checkpoint_path, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])
        return checkpoint_dict

    policy_id = getattr(cfg, "policy_index", 0)
    load_checkpoint_kind = getattr(cfg, "load_checkpoint_kind", "latest")
    name_prefix = dict(latest="checkpoint", best="best")[load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = load_sample_factory_checkpoint_from_paths(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])
    return checkpoint_dict


def _shape_from_observation_space(space: Any) -> tuple[int, ...] | None:
    if isinstance(space, Box):
        return tuple(int(dim) for dim in space.shape)
    if isinstance(space, SpaceDict) and "obs" in space.spaces:
        obs_space = space.spaces["obs"]
        if isinstance(obs_space, Box):
            return tuple(int(dim) for dim in obs_space.shape)
    return None


def validate_sample_factory_obs_shape(
    obs_dict: Mapping[str, torch.Tensor | np.ndarray],
    *,
    expected_shape: tuple[int, ...] | None,
    policy_type: str,
) -> None:
    if expected_shape is None:
        return
    obs = obs_dict.get("obs")
    if obs is None:
        return
    actual_shape = tuple(int(dim) for dim in obs.shape[-len(expected_shape) :])
    if actual_shape != tuple(int(dim) for dim in expected_shape):
        raise ValueError(
            f"{policy_type} observation shape mismatch: policy/checkpoint expects "
            f"{tuple(expected_shape)}, got {actual_shape}. Check env.obs_resize/env.obs_mode "
            "and retrain checkpoints when channel count or resize changes."
        )


class SampleFactoryPolicyRunner(PolicyRunner):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        action_resolver: ActionResolver,
        integration_module: Any,
        policy_type: str,
    ):
        from sample_factory.algo.sampling.batched_sampling import preprocess_actions
        from sample_factory.algo.utils.action_distributions import argmax_actions
        from sample_factory.algo.utils.env_info import extract_env_info
        from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
        from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
        from sample_factory.model.actor_critic import create_actor_critic
        from sample_factory.model.model_utils import get_rnn_size

        self.config = config
        self.action_resolver = action_resolver
        self.integration_module = integration_module
        self.policy_type = policy_type
        self.device = torch.device(_resolve_device(config["policy"].get("device", "auto")))
        self.deterministic = bool(config["policy"].get("deterministic", True))
        self._prepare_and_normalize_obs = prepare_and_normalize_obs
        self._argmax_actions = argmax_actions
        self._preprocess_actions = preprocess_actions
        self._unsqueeze_tensor = unsqueeze_tensor

        integration_module.register_components()
        cfg = integration_module.build_sf_cfg_from_config(config)
        self.cfg = cfg
        self.use_rnn = bool(getattr(cfg, "use_rnn", False))
        checkpoint_path = config["policy"].get("checkpoint_path") or None
        if checkpoint_path is None:
            from sample_factory.cfg.arguments import load_from_checkpoint

            cfg = load_from_checkpoint(cfg)
            self.cfg = cfg
            self.use_rnn = bool(getattr(cfg, "use_rnn", False))
        reference_env = integration_module.make_reference_env(cfg)
        self.env_info = extract_env_info(reference_env, cfg)
        self.action_space = reference_env.action_space
        self.actor_critic = create_actor_critic(cfg, reference_env.observation_space, reference_env.action_space)
        self.actor_critic.eval()
        self.actor_critic.model_to_device(self.device)

        self.expected_obs_shape = _shape_from_observation_space(reference_env.observation_space)
        try:
            checkpoint = _load_sf_policy_state_dict(cfg, self.actor_critic, self.device, checkpoint_path)
        except RuntimeError as exc:
            raise ValueError(
                f"{policy_type} checkpoint/model shape mismatch. Check env.obs_resize/env.obs_mode "
                "and use a checkpoint trained with the same observation shape."
            ) from exc
        self.checkpoint_global_step = int(checkpoint.get("train_step", checkpoint.get("env_steps", 0)))
        self.rnn_size = int(get_rnn_size(cfg))
        self.rnn_states = torch.zeros([1, self.rnn_size], dtype=torch.float32, device=self.device)
        self._rnn_states_by_slot: dict[int, torch.Tensor] = {}
        reference_env.close()

    def reset_state(self, slot_id: int | None = None) -> None:
        if slot_id is None:
            self.rnn_states.zero_()
            self._rnn_states_by_slot.clear()
            return
        self._rnn_states_by_slot[int(slot_id)] = self._zero_rnn_state()

    def predict(self, observation: Observation) -> PolicyOutput:
        if self.use_rnn and int(observation.env_step) == 0:
            self.rnn_states.zero_()
        rnn_states = self.rnn_states if self.use_rnn else self._zero_rnn_state()

        obs_dict, _action_mask = _obs_to_sf_inputs(observation, self.device)
        validate_sample_factory_obs_shape(
            obs_dict,
            expected_shape=self.expected_obs_shape,
            policy_type=self.policy_type,
        )
        normalized_obs = self._prepare_and_normalize_obs(self.actor_critic, obs_dict)
        with torch.no_grad():
            policy_outputs = self.actor_critic(normalized_obs, rnn_states)
            actions = policy_outputs["actions"]
            if self.deterministic:
                actions = self._deterministic_actions(batch_size=1)
            actions = _normalize_actions_for_space(
                actions,
                batch_size=1,
                action_space=self.action_space,
                device=self.device,
                unsqueeze_tensor=self._unsqueeze_tensor,
            )
            if self.use_rnn:
                self.rnn_states = policy_outputs["new_rnn_states"]

        processed_actions = self._preprocess_actions_for_space(actions)
        raw_value = _to_python_action(processed_actions)
        action = _resolve_action(raw_value, self.action_resolver)
        return PolicyOutput(
            action=action,
            raw_output=raw_value,
            metadata={
                "policy_type": self.policy_type,
                "checkpoint_global_step": self.checkpoint_global_step,
            },
        )

    def predict_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        observations = list(observations)
        if not observations:
            return []
        self._raise_if_batch_unsupported()

        obs_dict, _action_mask = _obs_batch_to_sf_inputs(observations, self.device)
        validate_sample_factory_obs_shape(
            obs_dict,
            expected_shape=self.expected_obs_shape,
            policy_type=self.policy_type,
        )
        normalized_obs = self._prepare_and_normalize_obs(self.actor_critic, obs_dict)
        batch_size = len(observations)
        slot_ids = None
        if self.use_rnn:
            slot_ids = self._slot_ids_for_observations(observations)
            rnn_states = torch.cat(
                [self._rnn_state_for_slot(slot_id) for slot_id in slot_ids],
                dim=0,
            )
        else:
            rnn_states = torch.zeros([batch_size, self.rnn_size], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_outputs = self.actor_critic(normalized_obs, rnn_states)
            actions = policy_outputs["actions"]
            if self.deterministic:
                actions = self._deterministic_actions(batch_size=batch_size)
            actions = _normalize_actions_for_space(
                actions,
                batch_size=batch_size,
                action_space=self.action_space,
                device=self.device,
                unsqueeze_tensor=self._unsqueeze_tensor,
            )
            if slot_ids is not None:
                self._store_batch_rnn_states(slot_ids, policy_outputs["new_rnn_states"])

        processed_actions = self._preprocess_actions_for_space(actions)
        raw_values = _split_batch_actions(processed_actions, batch_size, action_space=self.action_space)
        return [
            PolicyOutput(
                action=_resolve_action(raw_value, self.action_resolver),
                raw_output=raw_value,
                metadata={
                    "policy_type": self.policy_type,
                    "checkpoint_global_step": self.checkpoint_global_step,
                },
            )
            for raw_value in raw_values
        ]

    def _raise_if_batch_unsupported(self) -> None:
        if _is_supported_batched_policy_action_space(self.action_space):
            return
        if isinstance(self.action_space, SpaceTuple):
            raise NotImplementedError(
                "Phase 3D supports flat Tuple(Discrete, ...) action spaces only; "
                f"got action_space={self.action_space!r}"
            )
        if isinstance(self.action_space, MultiDiscrete):
            raise NotImplementedError(
                "Phase 3D supports one-dimensional MultiDiscrete action spaces only; "
                f"got nvec={self.action_space.nvec!r}"
            )
        raise NotImplementedError(
            "Sample Factory batched simulated eval supports Discrete, flat Tuple(Discrete, ...), "
            f"and one-dimensional MultiDiscrete action spaces; got action_space={self.action_space!r}"
        )

    def _zero_rnn_state(self) -> torch.Tensor:
        return torch.zeros([1, self.rnn_size], dtype=torch.float32, device=self.device)

    def _slot_ids_for_observations(self, observations: Sequence[Observation]) -> list[int]:
        slot_ids = []
        for observation in observations:
            slot_id = int(observation.metadata["slot_id"])
            if int(observation.env_step) == 0:
                self._rnn_states_by_slot[slot_id] = self._zero_rnn_state()
            slot_ids.append(slot_id)
        return slot_ids

    def _rnn_state_for_slot(self, slot_id: int) -> torch.Tensor:
        return self._rnn_states_by_slot[slot_id]

    def _store_batch_rnn_states(self, slot_ids: Sequence[int], new_rnn_states: torch.Tensor) -> None:
        for index, slot_id in enumerate(slot_ids):
            self._rnn_states_by_slot[int(slot_id)] = new_rnn_states[index : index + 1]

    def _deterministic_actions(self, *, batch_size: int) -> torch.Tensor:
        action_distribution = self.actor_critic.action_distribution()
        if _is_flat_discrete_tuple_space(self.action_space):
            distributions = getattr(action_distribution, "distributions", None)
            if distributions is None or len(distributions) != len(self.action_space.spaces):
                raise ValueError(
                    "Sample Factory Tuple action distribution must expose one distribution per action component"
                )
            component_actions = []
            for component_distribution in distributions:
                component = _as_action_tensor(self._argmax_actions(component_distribution), self.device)
                if component.ndim == 0:
                    component = component.reshape(1)
                if component.ndim == 2 and component.shape[-1] == 1:
                    component = component.squeeze(-1)
                if component.ndim != 1:
                    raise ValueError(
                        "Sample Factory Tuple argmax component must have shape [batch], "
                        f"got {tuple(component.shape)}"
                    )
                component_actions.append(component)
            actions = torch.stack(component_actions, dim=1)
            if int(actions.shape[0]) != int(batch_size):
                raise ValueError(
                    "Sample Factory Tuple argmax returned the wrong batch size: "
                    f"expected {batch_size}, got {actions.shape[0]}"
                )
            return actions
        return _as_action_tensor(self._argmax_actions(action_distribution), self.device)

    def _preprocess_actions_for_space(self, actions: torch.Tensor) -> Any:
        if isinstance(self.action_space, MultiDiscrete):
            return actions.detach().cpu().numpy()
        return self._preprocess_actions(self.env_info, actions)


def _is_flat_discrete_tuple_space(action_space: Any) -> bool:
    return isinstance(action_space, SpaceTuple) and all(
        isinstance(subspace, Discrete)
        for subspace in action_space.spaces
    )


def _is_one_dimensional_multidiscrete_space(action_space: Any) -> bool:
    return isinstance(action_space, MultiDiscrete) and np.asarray(action_space.nvec).ndim == 1


def _is_supported_batched_policy_action_space(action_space: Any) -> bool:
    return (
        isinstance(action_space, Discrete)
        or _is_flat_discrete_tuple_space(action_space)
        or _is_one_dimensional_multidiscrete_space(action_space)
    )


def _as_action_tensor(actions: Any, device: torch.device) -> torch.Tensor:
    if isinstance(actions, torch.Tensor):
        return actions.to(device=device)
    return torch.as_tensor(actions, device=device)


def _normalize_actions_for_space(
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
        if batch_size == 1 and int(actions.shape[0]) == int(num_components):
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
    expected_shape = (int(batch_size), int(num_components))
    if tuple(int(dim) for dim in actions.shape) != expected_shape:
        raise ValueError(f"Sample Factory actions must have shape {expected_shape}, got {tuple(actions.shape)}")


def _validate_discrete_tuple_action_values(actions: torch.Tensor, action_space: SpaceTuple) -> None:
    for component_index, subspace in enumerate(action_space.spaces):
        values = actions[:, component_index]
        if torch.any(values < 0) or torch.any(values >= int(subspace.n)):
            raise ValueError(
                f"Tuple action component {component_index} is outside Discrete({int(subspace.n)})"
            )


def _validate_multidiscrete_action_values(actions: torch.Tensor, action_space: MultiDiscrete) -> None:
    nvec = [int(value) for value in np.asarray(action_space.nvec).tolist()]
    for component_index, n in enumerate(nvec):
        values = actions[:, component_index]
        if torch.any(values < 0) or torch.any(values >= n):
            raise ValueError(
                f"MultiDiscrete action component {component_index} is outside range [0, {n})"
            )


def _split_batch_actions(processed_actions: Any, batch_size: int, *, action_space: Any = None) -> list[Any]:
    if isinstance(processed_actions, torch.Tensor):
        processed_actions = processed_actions.detach().cpu().numpy()
    if isinstance(processed_actions, np.ndarray):
        if processed_actions.ndim == 0:
            if batch_size != 1:
                raise ValueError(
                    "Sample Factory batched policy returned a scalar processed action for multiple observations"
                )
            return [_to_python_action(processed_actions)]
        if int(processed_actions.shape[0]) != batch_size:
            raise ValueError(
                "Sample Factory batched policy returned the wrong number of processed actions: "
                f"expected {batch_size}, got {processed_actions.shape[0]}"
            )
        return [_to_python_action(processed_actions[index]) for index in range(batch_size)]
    if isinstance(processed_actions, list):
        if _is_flat_discrete_tuple_space(action_space) and len(processed_actions) == len(action_space.spaces):
            return _split_tuple_component_batches(processed_actions, batch_size)
        if len(processed_actions) != batch_size:
            raise ValueError(
                "Sample Factory batched policy returned the wrong number of processed actions: "
                f"expected {batch_size}, got {len(processed_actions)}"
            )
        return [_to_python_action(item) for item in processed_actions]
    if batch_size == 1:
        return [_to_python_action(processed_actions)]
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
        if int(component.shape[0]) != int(batch_size):
            raise ValueError(
                "Sample Factory Tuple component returned the wrong batch size: "
                f"expected {batch_size}, got {component.shape[0]}"
            )
        components.append([_to_python_action(component[index]) for index in range(batch_size)])
    return [
        [_to_python_action(component_values[slot_index]) for component_values in components]
        for slot_index in range(batch_size)
    ]


def _to_python_action(value: Any):
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
            return _to_python_action(value[0])
        return type(value)(_to_python_action(item) for item in value)
    return value


def _resolve_action(raw_value: Any, action_resolver: ActionResolver) -> Action:
    if isinstance(raw_value, list):
        raw_value = tuple(raw_value)
    try:
        return action_resolver.resolve(raw_value)
    except Exception:
        return Action(value=raw_value, name=str(raw_value), is_noop=False, is_oneshot=False)
