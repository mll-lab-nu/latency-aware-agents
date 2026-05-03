from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import numpy as np
import numpy.typing as npt
import torch

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.factory import register_policy


OpenVlaActionOutput = npt.NDArray[np.floating[Any]] | torch.Tensor
OpenVlaPrediction = tuple[OpenVlaActionOutput, Mapping[str, Any]]


class OpenVlaEnvObs(TypedDict):
    main_images: torch.Tensor
    states: torch.Tensor
    task_descriptions: list[str]


@dataclass(frozen=True)
class OpenVlaPolicyConfig:
    base_model_path: Path
    checkpoint_path: Path
    prompt: str
    image_size: int
    action_dim: int
    num_action_chunks: int
    add_value_head: bool
    max_prompt_length: int
    num_images_in_input: int
    center_crop: bool
    trust_remote_code: bool
    unnorm_key: str
    device: torch.device


class OpenVlaLoaderConfig(dict[str, Any]):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


class OpenVlaModelProtocol(Protocol):
    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool) -> object:
        ...

    def to(self, device: torch.device) -> object:
        ...

    def eval(self) -> object:
        ...


class OpenVlaPredictProtocol(Protocol):
    def predict_action_batch(
        self,
        *,
        env_obs: OpenVlaEnvObs,
        calculate_logprobs: bool,
        calculate_values: bool,
        do_sample: bool,
        temperature: float,
        top_k: int,
    ) -> object:
        ...


class FlappyOpenVlaPolicyRunner(PolicyRunner):
    def __init__(
        self,
        model: object,
        config: OpenVlaPolicyConfig,
        device: torch.device,
        action_by_raw_id: Mapping[int, Action],
    ) -> None:
        self._model = model
        self._config = config
        self._device = device
        self._action_by_raw_id = dict(action_by_raw_id)

    def predict(self, observation: Observation) -> PolicyOutput:
        frame_hwc = observation_data_to_hwc_uint8(observation.data, self._config.image_size)
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
        action = self._action_by_raw_id.get(raw_action_id)
        if action is None:
            configured_ids = sorted(self._action_by_raw_id)
            raise ValueError(
                "OpenVLA emitted unmapped Flappy action id "
                f"{raw_action_id}; configured ids: {configured_ids}"
            )
        return PolicyOutput(
            action=action,
            raw_output=raw_action_id,
            metadata={
                "policy_type": "flappy_openvla",
                "base_model_path": str(self._config.base_model_path),
                "checkpoint_path": str(self._config.checkpoint_path),
                "prompt_sha256": sha256_text(self._config.prompt),
                "image_size": self._config.image_size,
                "action_dim": self._config.action_dim,
                "input_frame_sha256": sha256_array(frame_hwc),
                "input_frame_shape": list(frame_hwc.shape),
                "raw_action_head": raw_action_head,
                "raw_action_tokens": openvla_action_tokens_from_prediction(prediction),
                "unnorm_key": self._config.unnorm_key,
                "device": str(self._device),
            },
        )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_array(value: npt.NDArray[Any]) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def extract_openvla_policy_config(policy_cfg: Mapping[str, object]) -> OpenVlaPolicyConfig:
    device_name = _required_string(policy_cfg, "device")
    try:
        device = torch.device(device_name)
    except RuntimeError as error:
        raise ValueError(f"policy.device must be a valid torch device, got {device_name}.") from error
    return OpenVlaPolicyConfig(
        base_model_path=_required_path(policy_cfg, "base_model_path"),
        checkpoint_path=_required_path(policy_cfg, "checkpoint_path"),
        prompt=_required_string(policy_cfg, "prompt"),
        image_size=_required_positive_int(policy_cfg, "image_size"),
        action_dim=_required_positive_int(policy_cfg, "action_dim"),
        num_action_chunks=_required_positive_int(policy_cfg, "num_action_chunks"),
        add_value_head=_required_bool(policy_cfg, "add_value_head"),
        max_prompt_length=_required_positive_int(policy_cfg, "max_prompt_length"),
        num_images_in_input=_required_positive_int(policy_cfg, "num_images_in_input"),
        center_crop=_required_bool(policy_cfg, "center_crop"),
        trust_remote_code=_required_bool(policy_cfg, "trust_remote_code"),
        unnorm_key=_required_string(policy_cfg, "unnorm_key"),
        device=device,
    )


def openvla_loader_config(config: OpenVlaPolicyConfig) -> OpenVlaLoaderConfig:
    return OpenVlaLoaderConfig(
        {
            "model_path": str(config.base_model_path),
            "action_dim": config.action_dim,
            "num_action_chunks": config.num_action_chunks,
            "add_value_head": config.add_value_head,
            "max_prompt_length": config.max_prompt_length,
            "num_images_in_input": config.num_images_in_input,
            "center_crop": config.center_crop,
            "trust_remote_code": config.trust_remote_code,
            "unnorm_key": config.unnorm_key,
        }
    )


def load_openvla_sft_model(config: OpenVlaPolicyConfig) -> tuple[object, torch.device]:
    statistics_path = _require_file(config.base_model_path / "dataset_statistics.json")
    raw = statistics_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(
            f"dataset_statistics.json at {statistics_path} is empty. "
            "Populate it with a flappy_bird entry — see RL-inf ensure_dataset_statistics()."
        )
    statistics = json.loads(raw)
    if not isinstance(statistics, Mapping):
        raise TypeError(
            "Expected dataset statistics JSON object at "
            f"{statistics_path}, got {type(statistics).__name__}"
        )
    if config.unnorm_key not in statistics:
        raise KeyError(
            f"Missing unnorm key {config.unnorm_key} in dataset statistics at {statistics_path}"
        )
    weights_path = _require_file(_full_weights_path(config.checkpoint_path))
    try:
        from rlinf.models.embodiment.openvla_oft.rlinf import get_model
    except ImportError as error:
        raise ImportError(
            f"Could not import RL-inf OpenVLA get_model ({error}). "
            "Ensure PYTHONPATH includes the RL-inf root and all RL-inf dependencies "
            "(omegaconf, prismatic, transformers) are installed in the active env."
        ) from error

    print(
        "[DEBUG][latency_bench.openvla] Loading OpenVLA base model from "
        f"{config.base_model_path}",
        flush=True,
    )
    model = cast(OpenVlaModelProtocol, get_model(openvla_loader_config(config)))
    print(
        "[DEBUG][latency_bench.openvla] Loading OpenVLA SFT checkpoint from "
        f"{weights_path}",
        flush=True,
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"Expected mapping state_dict at {weights_path}, got {type(state_dict).__name__}"
        )
    tensor_state_dict = _tensor_state_dict(weights_path, state_dict)
    load_result = model.load_state_dict(tensor_state_dict, strict=True)
    missing_keys = getattr(load_result, "missing_keys", ())
    unexpected_keys = getattr(load_result, "unexpected_keys", ())
    print(
        "[DEBUG][latency_bench.openvla] Loaded OpenVLA SFT checkpoint "
        f"tensors={len(tensor_state_dict)} "
        f"missing_keys={len(missing_keys)} "
        f"unexpected_keys={len(unexpected_keys)}",
        flush=True,
    )
    model.to(config.device)
    model.eval()
    return model, config.device


@register_policy("flappy_openvla")
def _build_flappy_openvla_policy(
    config: dict[str, object],
    action_resolver: ActionResolver,
) -> PolicyRunner:
    policy_cfg = config["policy"]
    if not isinstance(policy_cfg, Mapping):
        raise TypeError(f"config.policy must be a mapping, got {type(policy_cfg).__name__}.")
    openvla_config = extract_openvla_policy_config(policy_cfg)
    action_by_raw_id = build_action_by_raw_id(policy_cfg, action_resolver, openvla_config.action_dim)
    model, device = load_openvla_sft_model(openvla_config)
    return FlappyOpenVlaPolicyRunner(model, openvla_config, device, action_by_raw_id)


def build_action_by_raw_id(
    policy_cfg: Mapping[str, object],
    action_resolver: ActionResolver,
    action_dim: int,
) -> dict[int, Action]:
    if "actions" not in policy_cfg:
        raise ValueError("policy.actions is required for flappy_openvla.")
    action_refs = policy_cfg["actions"]
    if not isinstance(action_refs, list):
        raise TypeError(f"policy.actions must be a list, got {type(action_refs).__name__}.")
    if len(action_refs) != action_dim:
        raise ValueError(
            f"policy.actions length must equal policy.action_dim {action_dim}, got {len(action_refs)}."
        )
    return {
        raw_action_id: action_resolver.resolve(action_ref)
        for raw_action_id, action_ref in enumerate(action_refs)
    }


def observation_data_to_hwc_uint8(data: Any, image_size: int) -> npt.NDArray[np.uint8]:
    from PIL import Image as _PILImage

    frame = np.asarray(data)
    if frame.ndim != 3:
        raise ValueError(f"Expected RGB image with rank 3, got shape {frame.shape}")

    if frame.shape == (image_size, image_size, 3):
        return _as_uint8_image(frame)
    if frame.shape == (3, image_size, image_size):
        return _as_uint8_image(np.transpose(frame, (1, 2, 0)))

    # Normalise to HWC first, then resize.
    if frame.shape[0] == 3 and frame.shape[1] != 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB image with 3 channels for Flappy observation, got shape {frame.shape}"
        )
    uint8_frame = _as_uint8_image(frame)
    resized = np.asarray(
        _PILImage.fromarray(uint8_frame).resize((image_size, image_size), _PILImage.BILINEAR),
        dtype=np.uint8,
    )
    return resized


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


def discrete_action_from_openvla_output(value: Any, action_dim: int) -> int:
    vector = _openvla_action_vector(value, action_dim)
    return int(np.argmax(vector))


def openvla_action_head(value: Any, action_dim: int) -> list[float]:
    vector = _openvla_action_vector(value, action_dim)
    return [float(item) for item in vector[:action_dim]]


def _openvla_action_vector(value: Any, action_dim: int) -> npt.NDArray[Any]:
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}")
    if isinstance(value, torch.Tensor):
        action_values = value.detach().cpu().numpy()
    else:
        action_values = np.asarray(value)
    if action_values.ndim == 3:
        vector = action_values[0, 0]
    elif action_values.ndim == 2:
        vector = action_values[0]
    elif action_values.ndim == 1:
        vector = action_values
    else:
        raise ValueError(f"Expected OpenVLA action output rank 1, 2, or 3, got {action_values.ndim}")
    if vector.shape[0] != action_dim:
        raise ValueError(f"Expected OpenVLA action_dim {action_dim}, got vector shape {vector.shape}")
    return vector


def openvla_action_from_prediction(prediction: Any) -> OpenVlaActionOutput:
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise TypeError(f"Expected OpenVLA prediction tuple of length 2, got {type(prediction).__name__}")
    typed_prediction = cast(OpenVlaPrediction, prediction)
    return typed_prediction[0]


def openvla_action_tokens_from_prediction(prediction: Any) -> list[int]:
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise TypeError(f"Expected OpenVLA prediction tuple of length 2, got {type(prediction).__name__}")
    metadata = prediction[1]
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Expected OpenVLA prediction metadata mapping, got {type(metadata).__name__}")
    forward_inputs = metadata.get("forward_inputs")
    if not isinstance(forward_inputs, Mapping):
        return []
    action_tokens = forward_inputs.get("action_tokens")
    if action_tokens is None:
        return []
    if isinstance(action_tokens, torch.Tensor):
        values = action_tokens.detach().cpu().numpy()
    else:
        values = np.asarray(action_tokens)
    if values.ndim == 3:
        vector = values[0, 0]
    elif values.ndim == 2:
        vector = values[0]
    elif values.ndim == 1:
        vector = values
    else:
        raise ValueError(f"Expected OpenVLA action_tokens rank 1, 2, or 3, got {values.ndim}")
    return [int(item) for item in vector.tolist()]


def build_openvla_env_obs(
    frame_hwc: npt.NDArray[np.uint8],
    prompt: str,
    device: torch.device,
) -> OpenVlaEnvObs:
    return {
        "main_images": torch.as_tensor(
            np.expand_dims(frame_hwc, axis=0),
            dtype=torch.uint8,
            device=device,
        ),
        "states": torch.zeros((1, 1), dtype=torch.float32, device=device),
        "task_descriptions": [prompt],
        "wrist_images": None,
        "extra_view_images": None,
    }


def _required_value(policy_cfg: Mapping[str, object], key: str) -> object:
    if key not in policy_cfg:
        raise ValueError(f"policy.{key} is required for flappy_openvla.")
    return policy_cfg[key]


def _required_string(policy_cfg: Mapping[str, object], key: str) -> str:
    value = _required_value(policy_cfg, key)
    if not isinstance(value, str):
        raise TypeError(f"policy.{key} must be str, got {type(value).__name__}.")
    if value == "":
        raise ValueError(f"policy.{key} must be a non-empty string for flappy_openvla.")
    return value


def _required_path(policy_cfg: Mapping[str, object], key: str) -> Path:
    value = _required_value(policy_cfg, key)
    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        raise TypeError(f"policy.{key} must be str or Path, got {type(value).__name__}.")
    if value == "":
        raise ValueError(f"policy.{key} must be a non-empty path for flappy_openvla.")
    return Path(value)


def _required_positive_int(policy_cfg: Mapping[str, object], key: str) -> int:
    value = _required_value(policy_cfg, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"policy.{key} must be int, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"policy.{key} must be positive for flappy_openvla, got {value}.")
    return value


def _required_bool(policy_cfg: Mapping[str, object], key: str) -> bool:
    value = _required_value(policy_cfg, key)
    if not isinstance(value, bool):
        raise TypeError(f"policy.{key} must be bool, got {type(value).__name__}.")
    return value


def _full_weights_path(checkpoint_path: Path) -> Path:
    return checkpoint_path / "actor" / "model_state_dict" / "full_weights.pt"


def _tensor_state_dict(weights_path: Path, state_dict: Mapping[object, object]) -> dict[str, torch.Tensor]:
    tensor_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Expected string state_dict key at {weights_path}, got {type(key).__name__}."
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Expected tensor state_dict value for {key} at {weights_path}, "
                f"got {type(value).__name__}."
            )
        tensor_state_dict[key] = value
    return tensor_state_dict


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return path


__all__ = [
    "FlappyOpenVlaPolicyRunner",
    "OpenVlaActionOutput",
    "OpenVlaEnvObs",
    "OpenVlaLoaderConfig",
    "OpenVlaPrediction",
    "OpenVlaPolicyConfig",
    "build_action_by_raw_id",
    "build_openvla_env_obs",
    "discrete_action_from_openvla_output",
    "extract_openvla_policy_config",
    "load_openvla_sft_model",
    "observation_data_to_hwc_uint8",
    "openvla_action_from_prediction",
    "openvla_action_tokens_from_prediction",
    "openvla_action_head",
    "openvla_loader_config",
    "sha256_array",
    "sha256_text",
]
