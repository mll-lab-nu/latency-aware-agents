from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.factory import register_policy
from latency_bench.policy.flappy_openvla import (
    OpenVlaPolicyConfig,
    OpenVlaPredictProtocol,
    build_openvla_env_obs,
    discrete_action_from_openvla_output,
    extract_openvla_policy_config,
    load_openvla_sft_model,
    openvla_action_from_prediction,
    openvla_action_head,
    openvla_action_tokens_from_prediction,
    sha256_array,
    sha256_text,
)


DemonAttackOpenVlaPolicyConfig = OpenVlaPolicyConfig

DEMON_ATTACK_OPENVLA_ACTION_LABELS: tuple[str, ...] = (
    "NOOP",
    "FIRE",
    "RIGHT",
    "LEFT",
    "RIGHTFIRE",
    "LEFTFIRE",
)
DEMON_ATTACK_OPENVLA_ACTION_DIM = 6


class DemonAttackOpenVlaPolicyRunner(PolicyRunner):
    def __init__(
        self,
        model: object,
        config: DemonAttackOpenVlaPolicyConfig,
        device: torch.device,
    ) -> None:
        self._model = model
        self._config = config
        self._device = device

    def predict(self, observation: Observation) -> PolicyOutput:
        frame_hwc = observation_data_to_hwc_rgb_uint8(observation.data, self._config.image_size)
        env_obs = build_openvla_env_obs(frame_hwc, self._config.prompt, self._device)
        model = cast(OpenVlaPredictProtocol, self._model)
        prediction = model.predict_action_batch(
            env_obs=env_obs,
            calculate_logprobs=False,
            calculate_values=False,
            do_sample=False,
            temperature=1.0,
            top_k=1,
        )
        raw_action = openvla_action_from_prediction(prediction)
        raw_action_id = discrete_action_from_openvla_output(raw_action, self._config.action_dim)
        raw_action_head = openvla_action_head(raw_action, self._config.action_dim)
        action = demon_attack_action_from_id(raw_action_id)
        return PolicyOutput(
            action=action,
            raw_output=raw_action_id,
            metadata={
                "policy_type": "demon_attack_openvla",
                "base_model_path": str(self._config.base_model_path),
                "checkpoint_path": str(self._config.checkpoint_path),
                "dataset_statistics_path": str(self._config.dataset_statistics_path),
                "prompt_sha256": sha256_text(self._config.prompt),
                "image_size": self._config.image_size,
                "action_dim": self._config.action_dim,
                "input_frame_sha256": sha256_array(frame_hwc),
                "input_frame_shape": list(frame_hwc.shape),
                "raw_action_head": raw_action_head,
                "raw_action_tokens": openvla_action_tokens_from_prediction(prediction),
                "unnorm_key": self._config.unnorm_key,
                "device": str(self._device),
                "action_label": action.name,
            },
        )


def demon_attack_action_from_id(action_id: int) -> Action:
    if action_id < 0 or action_id >= DEMON_ATTACK_OPENVLA_ACTION_DIM:
        raise ValueError(
            f"Demon Attack OpenVLA emitted action id {action_id}; "
            f"expected 0..{DEMON_ATTACK_OPENVLA_ACTION_DIM - 1}"
        )
    return Action(
        value=int(action_id),
        name=DEMON_ATTACK_OPENVLA_ACTION_LABELS[action_id],
        is_noop=action_id == 0,
        is_oneshot=False,
    )


def observation_data_to_hwc_rgb_uint8(
    data: Any,
    image_size: int | None = None,
) -> npt.NDArray[np.uint8]:
    from PIL import Image as _PILImage

    frame = np.asarray(data)
    if frame.ndim == 2:
        frame = np.expand_dims(frame, axis=-1)
    if frame.ndim != 3:
        raise ValueError(f"Expected Demon Attack image rank 3, got shape {frame.shape}")

    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[0] == 1:
        frame = np.repeat(np.transpose(frame, (1, 2, 0)), 3, axis=-1)
    elif frame.shape[-1] == 3:
        pass
    elif frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    else:
        raise ValueError(
            "Expected Demon Attack observation as grayscale HWC/CHW or RGB HWC/CHW, "
            f"got shape {frame.shape}"
        )

    uint8_frame = _as_uint8_image(frame)
    if image_size is not None and uint8_frame.shape[:2] != (image_size, image_size):
        uint8_frame = np.asarray(
            _PILImage.fromarray(uint8_frame).resize((image_size, image_size), _PILImage.BILINEAR),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(uint8_frame)


def _as_uint8_image(frame: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)
    if not np.issubdtype(frame.dtype, np.integer):
        raise TypeError(f"Expected integer image dtype, got {frame.dtype}")
    min_value = int(np.min(frame))
    max_value = int(np.max(frame))
    if min_value < 0 or max_value > 255:
        raise ValueError(
            f"Expected integer image values in [0, 255], got min {min_value} max {max_value}"
        )
    return np.ascontiguousarray(frame, dtype=np.uint8)


@register_policy("demon_attack_openvla")
def _build_demon_attack_openvla_policy(
    config: dict[str, object],
    action_resolver: ActionResolver,
) -> PolicyRunner:
    del action_resolver
    policy_cfg = config["policy"]
    if not isinstance(policy_cfg, Mapping):
        raise TypeError(f"config.policy must be a mapping, got {type(policy_cfg).__name__}.")
    openvla_config = extract_openvla_policy_config(policy_cfg)
    _validate_policy_config(policy_cfg, openvla_config)
    model, device = load_openvla_sft_model(openvla_config)
    return DemonAttackOpenVlaPolicyRunner(model, openvla_config, device)


def _validate_policy_config(
    policy_cfg: Mapping[str, object],
    openvla_config: DemonAttackOpenVlaPolicyConfig,
) -> None:
    if openvla_config.action_dim != DEMON_ATTACK_OPENVLA_ACTION_DIM:
        raise ValueError(
            "demon_attack_openvla requires policy.action_dim "
            f"{DEMON_ATTACK_OPENVLA_ACTION_DIM}, got {openvla_config.action_dim}"
        )
    configured_actions = policy_cfg.get("actions")
    if configured_actions is not None and list(configured_actions) != list(DEMON_ATTACK_OPENVLA_ACTION_LABELS):
        raise ValueError(
            "demon_attack_openvla policy.actions must be "
            f"{list(DEMON_ATTACK_OPENVLA_ACTION_LABELS)}, got {configured_actions}"
        )


__all__ = [
    "DEMON_ATTACK_OPENVLA_ACTION_DIM",
    "DEMON_ATTACK_OPENVLA_ACTION_LABELS",
    "DemonAttackOpenVlaPolicyConfig",
    "DemonAttackOpenVlaPolicyRunner",
    "demon_attack_action_from_id",
    "observation_data_to_hwc_rgb_uint8",
]
