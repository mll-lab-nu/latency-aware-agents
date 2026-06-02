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
    extract_openvla_policy_config,
    load_openvla_sft_model,
    openvla_action_from_prediction,
    openvla_action_head,
    openvla_action_tokens_from_prediction,
    sha256_array,
    sha256_text,
)
from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY


DeadlyCorridorOpenVlaPolicyConfig = OpenVlaPolicyConfig

DEADLY_CORRIDOR_OPENVLA_ACTION_LABELS: tuple[str, ...] = (
    "TURN_NOOP",
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_NOOP",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "STRAFE_NOOP",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "ATTACK_NOOP",
    "ATTACK",
)
DEADLY_CORRIDOR_OPENVLA_ACTION_LAYOUT = "deadly_corridor_factorized_11"
DEADLY_CORRIDOR_OPENVLA_ACTION_DIM = 11
DEADLY_CORRIDOR_TURN_LABELS: tuple[str, ...] = (
    "TURN_NOOP",
    "TURN_LEFT",
    "TURN_RIGHT",
)
DEADLY_CORRIDOR_MOVE_LABELS: tuple[str, ...] = (
    "MOVE_NOOP",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
)
DEADLY_CORRIDOR_STRAFE_LABELS: tuple[str, ...] = (
    "STRAFE_NOOP",
    "MOVE_LEFT",
    "MOVE_RIGHT",
)
DEADLY_CORRIDOR_ATTACK_LABELS: tuple[str, ...] = (
    "ATTACK_NOOP",
    "ATTACK",
)


class DeadlyCorridorOpenVlaPolicyRunner(PolicyRunner):
    def __init__(
        self,
        model: object,
        config: DeadlyCorridorOpenVlaPolicyConfig,
        device: torch.device,
    ) -> None:
        self._model = model
        self._config = config
        self._device = device

    def predict(self, observation: Observation) -> PolicyOutput:
        frame_source = observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]
        frame_hwc = observation_data_to_single_hwc_rgb_uint8(
            frame_source,
            image_size=self._config.image_size,
        )
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
        raw_action_head = openvla_action_head(raw_action, self._config.action_dim)
        action, action_metadata = deadly_corridor_action_from_factorized_vector(raw_action_head)
        return PolicyOutput(
            action=action,
            raw_output=list(action.value),
            metadata={
                "policy_type": "deadly_corridor_openvla",
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
                "action_layout": DEADLY_CORRIDOR_OPENVLA_ACTION_LAYOUT,
                **action_metadata,
            },
        )


def deadly_corridor_action_from_factorized_vector(values: Any) -> tuple[Action, dict[str, object]]:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"Expected Deadly Corridor factorized action rank 1, got shape {vector.shape}")
    if vector.shape[0] != DEADLY_CORRIDOR_OPENVLA_ACTION_DIM:
        raise ValueError(
            "Deadly Corridor OpenVLA emitted action vector with shape "
            f"{vector.shape}; expected ({DEADLY_CORRIDOR_OPENVLA_ACTION_DIM},)"
        )
    turn = int(np.argmax(vector[0:3]))
    move = int(np.argmax(vector[3:6]))
    strafe = int(np.argmax(vector[6:9]))
    attack = int(np.argmax(vector[9:11]))
    action_value = [turn, move, strafe, attack]
    turn_label = DEADLY_CORRIDOR_TURN_LABELS[turn]
    move_label = DEADLY_CORRIDOR_MOVE_LABELS[move]
    strafe_label = DEADLY_CORRIDOR_STRAFE_LABELS[strafe]
    attack_label = DEADLY_CORRIDOR_ATTACK_LABELS[attack]
    active_labels = [
        label
        for label in (turn_label, move_label, strafe_label, attack_label)
        if not label.endswith("_NOOP")
    ]
    action_name = "+".join(active_labels) if active_labels else "NOOP"
    return Action(
        value=action_value,
        name=action_name,
        is_noop=action_value == [0, 0, 0, 0],
        is_oneshot=False,
    ), {
        "decoded_action_tuple": action_value,
        "turn_label": turn_label,
        "move_label": move_label,
        "strafe_label": strafe_label,
        "attack_label": attack_label,
    }


def observation_data_to_single_hwc_rgb_uint8(
    data: Any,
    image_size: int | None = None,
) -> npt.NDArray[np.uint8]:
    from PIL import Image as _PILImage

    frame = _extract_observation_array(data)
    if frame.ndim == 4 and frame.shape[-1] == 3:
        frame = frame[-1]
    elif frame.ndim == 4 and frame.shape[1] == 3:
        frame = np.transpose(frame[-1], (1, 2, 0))
    if frame.ndim != 3:
        raise ValueError(f"Expected Deadly Corridor image rank 3, got shape {frame.shape}")

    if frame.shape[-1] == 3:
        rgb_hwc = frame
    elif frame.shape[0] == 3:
        rgb_hwc = np.transpose(frame, (1, 2, 0))
    elif frame.shape[0] > 3 and frame.shape[0] % 3 == 0:
        latest_chw = frame[-3:, :, :]
        rgb_hwc = np.transpose(latest_chw, (1, 2, 0))
    elif frame.shape[-1] > 3 and frame.shape[-1] % 3 == 0:
        rgb_hwc = frame[:, :, -3:]
    else:
        raise ValueError(
            "Expected Deadly Corridor observation as RGB HWC, RGB CHW, or stacked CHW, "
            f"got shape {frame.shape}"
        )

    uint8_frame = _as_uint8_image(rgb_hwc)
    if image_size is not None and uint8_frame.shape[:2] != (image_size, image_size):
        uint8_frame = np.asarray(
            _PILImage.fromarray(uint8_frame).resize((image_size, image_size), _PILImage.BILINEAR),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(uint8_frame)


def _extract_observation_array(data: Any) -> npt.NDArray[Any]:
    if isinstance(data, Mapping):
        for key in ("obs", "image", "rgb", "screen", "frame", "observation"):
            if key in data:
                return np.asarray(data[key])
        if len(data) == 1:
            return np.asarray(next(iter(data.values())))
        raise ValueError(
            "Deadly Corridor dict observation must contain one of "
            "['obs', 'image', 'rgb', 'screen', 'frame', 'observation']"
        )
    return np.asarray(data)


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


@register_policy("deadly_corridor_openvla")
def _build_deadly_corridor_openvla_policy(
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
    return DeadlyCorridorOpenVlaPolicyRunner(model, openvla_config, device)


def _validate_policy_config(
    policy_cfg: Mapping[str, object],
    openvla_config: DeadlyCorridorOpenVlaPolicyConfig,
) -> None:
    if openvla_config.action_dim != DEADLY_CORRIDOR_OPENVLA_ACTION_DIM:
        raise ValueError(
            "deadly_corridor_openvla requires policy.action_dim "
            f"{DEADLY_CORRIDOR_OPENVLA_ACTION_DIM}, got {openvla_config.action_dim}"
        )
    action_layout = policy_cfg.get("action_layout", DEADLY_CORRIDOR_OPENVLA_ACTION_LAYOUT)
    if action_layout != DEADLY_CORRIDOR_OPENVLA_ACTION_LAYOUT:
        raise ValueError(
            "deadly_corridor_openvla requires policy.action_layout "
            f"{DEADLY_CORRIDOR_OPENVLA_ACTION_LAYOUT!r}, got {action_layout!r}"
        )
    configured_actions = policy_cfg.get("actions")
    if configured_actions is not None and list(configured_actions) != list(DEADLY_CORRIDOR_OPENVLA_ACTION_LABELS):
        raise ValueError(
            "deadly_corridor_openvla policy.actions must be "
            f"{list(DEADLY_CORRIDOR_OPENVLA_ACTION_LABELS)}, got {configured_actions}"
        )


__all__ = [
    "DEADLY_CORRIDOR_OPENVLA_ACTION_DIM",
    "DEADLY_CORRIDOR_OPENVLA_ACTION_LAYOUT",
    "DEADLY_CORRIDOR_OPENVLA_ACTION_LABELS",
    "DeadlyCorridorOpenVlaPolicyConfig",
    "DeadlyCorridorOpenVlaPolicyRunner",
    "deadly_corridor_action_from_factorized_vector",
    "observation_data_to_single_hwc_rgb_uint8",
]
