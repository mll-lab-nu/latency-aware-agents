"""Build and run Sample Factory policies through the benchmark interface."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from gymnasium.spaces import Box, Dict as SpaceDict, MultiDiscrete, Tuple as SpaceTuple

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Observation, PolicyOutput
from latency_bench.integrations.sample_factory import actions as sf_actions
from latency_bench.integrations.sample_factory import runtime as sf_runtime
from latency_bench.policy.base import PolicyRunner
from latency_bench.envs.factory import environment_integration
from latency_bench.policy.model_source import model_file


_INTEGRATIONS = {
    "flappy": ("training.flappy_sf.integration", "register_flappy_envs"),
    "demon_attack": (
        "training.demon_attack_sf.integration",
        "register_demon_attack_envs",
    ),
    "deadly_corridor": (
        "training.deadly_corridor_sf.integration",
        "register_deadly_corridor_components",
    ),
    "defend_the_line": (
        "training.deadly_corridor_sf.integration",
        "register_deadly_corridor_components",
    ),
    "gymnasium": (
        "training.gymnasium_sf.integration",
        "register_gymnasium_envs",
    ),
}


class SampleFactoryIntegrationModule:
    """Adapt a training-side integration module to the policy runner interface."""

    def __init__(self, module: Any, register_attr: str):
        self.module = module
        self._register_attr = register_attr

    def register_components(self) -> None:
        getattr(self.module, self._register_attr)()

    def build_sf_cfg_from_config(self, config: dict[str, Any]):
        return self.module.SAMPLE_FACTORY_CONFIG_PARSER.parse_eval(
            self.module.build_cli_args_from_config(config)
        )

    def make_reference_env(self, cfg: Any):
        return self.module.make_reference_env(cfg)


class _GymnasiumPolicyActionResolver:
    def __init__(self, action_resolver: ActionResolver, action_order: list[Any]):
        self._action_resolver = action_resolver
        self._action_order = action_order

    def resolve(self, policy_action_id: Any):
        return self._action_resolver.resolve(self._action_order[policy_action_id])


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
            key: torch.cat(tensors, dim=0) for key, tensors in batched_parts.items()
        }
    except RuntimeError as exc:
        raise ValueError(
            "Cannot batch Sample Factory observations with incompatible tensor shapes"
        ) from exc

    if all(mask is not None for mask in action_masks):
        return batched_obs, torch.cat(
            [mask for mask in action_masks if mask is not None], dim=0
        )
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
    tensors = []
    for observation in observations:
        data = observation.data
        if isinstance(data, Mapping):
            return None
        if isinstance(data, torch.Tensor):
            tensors.append(data)
            continue
        values.append(data)
    if tensors:
        if values:
            return None
        return {"obs": torch.stack(tensors, dim=0).to(device=device)}, None
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
            str(key): value for key, value in data.items() if str(key) != "action_mask"
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
        if (
            array.shape != first_shape
            or array.dtype != first_dtype
            or array.dtype == object
        ):
            return None
    return np.stack(arrays, axis=0)


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
) -> None:
    if expected_shape is None:
        return
    obs = obs_dict.get("obs")
    if obs is None:
        return
    actual_shape = tuple(int(dim) for dim in obs.shape[-len(expected_shape) :])
    if actual_shape != tuple(int(dim) for dim in expected_shape):
        raise ValueError(
            "Sample Factory observation shape mismatch: policy/checkpoint expects "
            f"{tuple(expected_shape)}, got {actual_shape}. Check env.obs_resize/env.obs_mode "
            "and retrain checkpoints when channel count or resize changes."
        )


class SampleFactoryPolicyRunner(PolicyRunner):
    """Run a checkpoint with the registered task's observation and action contract."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        action_resolver: ActionResolver,
        integration_module: Any,
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
        self.device = torch.device(
            sf_runtime.resolve_torch_device(config["executor"]["inference_devices"][0])
        )
        self.deterministic = config["policy"]["deterministic"]
        self._prepare_and_normalize_obs = prepare_and_normalize_obs
        self._argmax_actions = argmax_actions
        self._preprocess_actions = preprocess_actions
        self._unsqueeze_tensor = unsqueeze_tensor

        integration_module.register_components()
        cfg = integration_module.build_sf_cfg_from_config(config)
        self.cfg = cfg
        self.use_rnn = bool(getattr(cfg, "use_rnn", False))
        checkpoint_path = (
            model_file(config["policy"], "checkpoint")
            if config["policy"].get("checkpoint") is not None
            else None
        )
        if checkpoint_path is None:
            from sample_factory.cfg.arguments import load_from_checkpoint

            cfg = load_from_checkpoint(cfg)
            self.cfg = cfg
            self.use_rnn = bool(getattr(cfg, "use_rnn", False))
        self.action_chunk_horizon = cfg.fasttd3_action_chunk_horizon
        reference_env = integration_module.make_reference_env(cfg)
        self.env_info = extract_env_info(reference_env, cfg)
        self.action_space = reference_env.action_space
        self.actor_critic = create_actor_critic(
            cfg, reference_env.observation_space, reference_env.action_space
        )
        self.actor_critic.eval()
        self.actor_critic.model_to_device(self.device)

        self.expected_obs_shape = _shape_from_observation_space(
            reference_env.observation_space
        )
        try:
            checkpoint, resolved_checkpoint_path = (
                sf_runtime.load_sample_factory_policy_state_dict(
                    cfg,
                    self.actor_critic,
                    self.device,
                    checkpoint_path,
                )
            )
        except RuntimeError as exc:
            raise ValueError(
                "Sample Factory checkpoint/model shape mismatch. Check env.obs_resize/env.obs_mode "
                "and use a checkpoint trained with the same observation shape."
            ) from exc
        self.checkpoint_path = str(resolved_checkpoint_path)
        config["policy"].update(
            source="local",
            root=str(Path(self.checkpoint_path).parent),
            checkpoint=Path(self.checkpoint_path).name,
        )
        self.checkpoint_global_step = int(
            checkpoint.get("train_step", checkpoint.get("env_steps", 0))
        )
        self.rnn_size = int(get_rnn_size(cfg))
        self.rnn_states = torch.zeros(
            [1, self.rnn_size], dtype=torch.float32, device=self.device
        )
        self._rnn_states_by_slot: dict[int, torch.Tensor] = {}
        reference_env.close()

    def reset_state(self, slot_id: int | None = None) -> None:
        if slot_id is None:
            self.rnn_states.zero_()
            self._rnn_states_by_slot.clear()
            return
        self._rnn_states_by_slot[int(slot_id)] = self._zero_rnn_state()

    def predict(self, observation: Observation) -> PolicyOutput:
        if self.use_rnn and observation.env_step == 0:
            self.rnn_states.zero_()
        rnn_states = self.rnn_states if self.use_rnn else self._zero_rnn_state()

        obs_dict, _action_mask = _obs_to_sf_inputs(observation, self.device)
        validate_sample_factory_obs_shape(
            obs_dict,
            expected_shape=self.expected_obs_shape,
        )
        normalized_obs = self._prepare_and_normalize_obs(self.actor_critic, obs_dict)
        with torch.no_grad():
            policy_outputs = self.actor_critic(normalized_obs, rnn_states)
            actions = policy_outputs["actions"]
            if self.deterministic:
                actions = self._deterministic_actions()
            actions = sf_actions.normalize_actions_for_space(
                actions,
                batch_size=1,
                action_space=self.action_space,
                device=self.device,
                unsqueeze_tensor=self._unsqueeze_tensor,
            )
            if self.use_rnn:
                self.rnn_states = policy_outputs["new_rnn_states"]

        processed_actions = self._preprocess_actions_for_space(actions)
        raw_value = sf_actions.to_python_action(processed_actions)
        return self._policy_output(raw_value)

    def predict_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        observations = list(observations)
        if not observations:
            return []
        self._raise_if_batch_unsupported()

        obs_dict, _action_mask = _obs_batch_to_sf_inputs(observations, self.device)
        validate_sample_factory_obs_shape(
            obs_dict,
            expected_shape=self.expected_obs_shape,
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
            rnn_states = torch.zeros(
                [batch_size, self.rnn_size], dtype=torch.float32, device=self.device
            )
        with torch.no_grad():
            policy_outputs = self.actor_critic(normalized_obs, rnn_states)
            actions = policy_outputs["actions"]
            if self.deterministic:
                actions = self._deterministic_actions()
            actions = sf_actions.normalize_actions_for_space(
                actions,
                batch_size=batch_size,
                action_space=self.action_space,
                device=self.device,
                unsqueeze_tensor=self._unsqueeze_tensor,
            )
            if slot_ids is not None:
                self._store_batch_rnn_states(slot_ids, policy_outputs["new_rnn_states"])

        processed_actions = self._preprocess_actions_for_space(actions)
        raw_values = sf_actions.split_batch_actions(
            processed_actions,
            batch_size,
            action_space=self.action_space,
        )
        return [self._policy_output(raw_value) for raw_value in raw_values]

    def _policy_output(self, raw_value: Any) -> PolicyOutput:
        metadata = {
            "policy_type": "sample_factory",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_global_step": self.checkpoint_global_step,
        }
        if (
            self.cfg.algo == "FAST_TD3"
            and environment_integration(self.config) == "gymnasium"
        ):
            raw_chunk = np.asarray(raw_value, dtype=np.float32).reshape(
                self.action_chunk_horizon, -1
            )
            native_space = self.config["env"]["action_space"]
            native_chunk = sf_actions.scale_box_action(
                raw_chunk,
                np.asarray(native_space["low"], dtype=np.float32),
                np.asarray(native_space["high"], dtype=np.float32),
            )
            actions = [
                sf_actions.resolve_action(
                    row.tolist(), self.action_resolver, action_space=self.action_space
                )
                for row in native_chunk
            ]
            return PolicyOutput(
                action=actions[0],
                action_chunk=actions if self.action_chunk_horizon > 1 else None,
                raw_output=raw_chunk.tolist()
                if self.action_chunk_horizon > 1
                else raw_chunk[0].tolist(),
                metadata=metadata,
            )
        return PolicyOutput(
            action=sf_actions.resolve_action(
                raw_value, self.action_resolver, action_space=self.action_space
            ),
            raw_output=raw_value,
            metadata=metadata,
        )

    def _raise_if_batch_unsupported(self) -> None:
        if sf_actions.is_supported_batched_policy_action_space(self.action_space):
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
            "Sample Factory batched simulated eval supports Box, Discrete, flat Tuple(Discrete, ...), "
            f"and one-dimensional MultiDiscrete action spaces; got action_space={self.action_space!r}"
        )

    def _zero_rnn_state(self) -> torch.Tensor:
        return torch.zeros([1, self.rnn_size], dtype=torch.float32, device=self.device)

    def _slot_ids_for_observations(
        self, observations: Sequence[Observation]
    ) -> list[int]:
        slot_ids = []
        for observation in observations:
            slot_id = int(observation.metadata["slot_id"])
            if observation.env_step == 0:
                self._rnn_states_by_slot[slot_id] = self._zero_rnn_state()
            slot_ids.append(slot_id)
        return slot_ids

    def _rnn_state_for_slot(self, slot_id: int) -> torch.Tensor:
        return self._rnn_states_by_slot[slot_id]

    def _store_batch_rnn_states(
        self, slot_ids: Sequence[int], new_rnn_states: torch.Tensor
    ) -> None:
        for index, slot_id in enumerate(slot_ids):
            self._rnn_states_by_slot[int(slot_id)] = new_rnn_states[index : index + 1]

    def _deterministic_actions(self) -> torch.Tensor:
        return sf_actions.deterministic_actions_for_space(
            self.actor_critic.action_distribution(),
            action_space=self.action_space,
            argmax_actions=self._argmax_actions,
            device=self.device,
        )

    def _preprocess_actions_for_space(self, actions: torch.Tensor) -> Any:
        if isinstance(self.action_space, MultiDiscrete):
            return actions.detach().cpu().numpy()
        return self._preprocess_actions(self.env_info, actions)


def build_sample_factory_policy(
    config: dict[str, Any],
    action_resolver: ActionResolver,
) -> SampleFactoryPolicyRunner:
    integration = environment_integration(config)
    integration_module, register_components = _INTEGRATIONS[integration]
    policy_action_resolver = (
        _GymnasiumPolicyActionResolver(action_resolver, config["env"]["action_order"])
        if integration == "gymnasium" and "action_order" in config["env"]
        else action_resolver
    )
    return SampleFactoryPolicyRunner(
        config=config,
        action_resolver=policy_action_resolver,
        integration_module=SampleFactoryIntegrationModule(
            import_module(integration_module), register_components
        ),
    )
