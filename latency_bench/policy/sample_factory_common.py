from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from gymnasium.spaces import Box, Dict as SpaceDict

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.policy.base import PolicyRunner
from training.sample_factory_common.checkpoint_io import load_sample_factory_checkpoint_from_paths, trusted_torch_load


def _resolve_device(device: Any) -> str:
    device_name = str(device)
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
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


def _extract_action_mask(observation: Observation, device: torch.device):
    metadata = getattr(observation, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    action_mask = metadata.get("action_mask")
    if action_mask is None:
        return None
    return torch.as_tensor(action_mask, device=device).unsqueeze(0)


def _checkpoint_state_dict_from_path(path: str | Path, device: torch.device) -> dict[str, Any]:
    return trusted_torch_load(path, device)


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
        checkpoint_path = config["policy"].get("checkpoint_path") or None
        if checkpoint_path is None:
            from sample_factory.cfg.arguments import load_from_checkpoint

            cfg = load_from_checkpoint(cfg)
        reference_env = integration_module.make_reference_env(cfg)
        self.env_info = extract_env_info(reference_env, cfg)
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
        rnn_size = int(get_rnn_size(cfg))
        self.rnn_states = torch.zeros([1, rnn_size], dtype=torch.float32, device=self.device)
        reference_env.close()

    def predict(self, observation: Observation) -> PolicyOutput:
        if int(observation.env_step) == 0:
            self.rnn_states.zero_()

        obs_dict, _action_mask = _obs_to_sf_inputs(observation, self.device)
        validate_sample_factory_obs_shape(
            obs_dict,
            expected_shape=self.expected_obs_shape,
            policy_type=self.policy_type,
        )
        normalized_obs = self._prepare_and_normalize_obs(self.actor_critic, obs_dict)
        with torch.no_grad():
            policy_outputs = self.actor_critic(normalized_obs, self.rnn_states)
            actions = policy_outputs["actions"]
            if self.deterministic:
                actions = self._argmax_actions(self.actor_critic.action_distribution())
            if actions.ndim == 1:
                actions = self._unsqueeze_tensor(actions, dim=-1)
            self.rnn_states = policy_outputs["new_rnn_states"]

        processed_actions = self._preprocess_actions(self.env_info, actions)
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


def _to_python_action(value: Any):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
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
    try:
        return action_resolver.resolve(raw_value)
    except Exception:
        if isinstance(raw_value, list):
            raw_value = tuple(raw_value)
        return Action(value=raw_value, name=str(raw_value), is_noop=False, is_oneshot=False)
