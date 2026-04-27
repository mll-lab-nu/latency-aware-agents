from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from modeling.flappy_ppo.observation import FlappyPpoObservationConfig


def resize_frame_pil(
    frame: np.ndarray,
    image_size: int = 84,
    grayscale: bool = False,
) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    image = Image.fromarray(arr)
    image = image.convert("L" if grayscale else "RGB")
    target_size = (int(image_size), int(image_size))
    if image.size != target_size:
        image = image.resize(target_size, Image.BILINEAR)

    out = np.asarray(image, dtype=np.uint8)
    if grayscale:
        out = out[..., None]
    return out


def latency_channel_value(frame_skip: int | float, max_frame_skip: int) -> np.uint8:
    max_skip = max(1, int(max_frame_skip))
    value = min(max(float(frame_skip), 0.0) / float(max_skip), 1.0)
    return np.uint8(np.rint(value * 255.0))


def latency_channel(
    height: int,
    width: int,
    frame_skip: int | float,
    max_frame_skip: int,
) -> np.ndarray:
    value = latency_channel_value(frame_skip, max_frame_skip)
    return np.full((int(height), int(width), 1), value, dtype=np.uint8)


def append_latency_channel(
    stacked_obs: np.ndarray,
    frame_skip: int | float,
    max_frame_skip: int,
) -> np.ndarray:
    obs = np.asarray(stacked_obs, dtype=np.uint8)
    return np.concatenate(
        [
            obs,
            latency_channel(
                obs.shape[0],
                obs.shape[1],
                frame_skip=frame_skip,
                max_frame_skip=max_frame_skip,
            ),
        ],
        axis=2,
    )


def reset_frame_stack(frame_obs: np.ndarray, stack_frames: int) -> np.ndarray:
    frame = np.asarray(frame_obs, dtype=np.uint8)
    return np.concatenate([frame.copy() for _ in range(max(1, int(stack_frames)))], axis=2)


def step_frame_stack(
    stacked_obs: np.ndarray,
    frame_obs: np.ndarray,
    base_channels: int | None = None,
    reset: bool = False,
) -> np.ndarray:
    frame = np.asarray(frame_obs, dtype=np.uint8)
    if reset:
        if base_channels is None:
            channels = int(np.asarray(stacked_obs).shape[2])
            base_channels = frame.shape[2]
            stack_frames = max(1, channels // base_channels)
        else:
            stack_frames = max(1, int(np.asarray(stacked_obs).shape[2]) // int(base_channels))
        return reset_frame_stack(frame, stack_frames)

    stack = np.asarray(stacked_obs, dtype=np.uint8)
    channels = int(base_channels or frame.shape[2])
    if stack.shape[:2] != frame.shape[:2]:
        raise ValueError("stacked_obs and frame_obs must have matching height/width")
    if stack.shape[2] % channels != 0:
        raise ValueError("stacked_obs channel count must be divisible by base_channels")
    if stack.shape[2] == channels:
        return frame.copy()
    return np.concatenate([stack[:, :, channels:].copy(), frame.copy()], axis=2)


def build_stacked_observation(
    stacked_obs: np.ndarray,
    observation_config: FlappyPpoObservationConfig,
    frame_skip: int | float = 0,
) -> np.ndarray:
    if observation_config.obs_mode == "image_only":
        return np.asarray(stacked_obs, dtype=np.uint8)
    return append_latency_channel(
        stacked_obs,
        frame_skip=frame_skip,
        max_frame_skip=observation_config.max_frame_skip,
    )


def hwc_to_nchw_tensor(
    obs: np.ndarray,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    arr = np.asarray(obs, dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError("obs must be HWC")
    tensor = torch.as_tensor(arr, dtype=torch.uint8, device=device)
    return tensor.permute(2, 0, 1).unsqueeze(0).contiguous()


class FlappyPpoEvalObservationPipeline:
    """Single-environment HWC frame stacker for latency_bench policy evaluation."""

    def __init__(self, observation_config: FlappyPpoObservationConfig):
        self.observation_config = observation_config
        self._stack: np.ndarray | None = None
        self._last_observation: np.ndarray | None = None

    @property
    def current_frame_stack(self) -> np.ndarray | None:
        return None if self._stack is None else self._stack.copy()

    @property
    def current_observation(self) -> np.ndarray | None:
        return None if self._last_observation is None else self._last_observation.copy()

    def reset(self, frame_or_observation: Any, frame_skip: int | float = 0) -> np.ndarray:
        frame = self._frame_obs(frame_or_observation)
        self._stack = reset_frame_stack(frame, self.observation_config.stack_frames)
        self._last_observation = build_stacked_observation(
            self._stack,
            self.observation_config,
            frame_skip=frame_skip,
        )
        return self._last_observation.copy()

    def step(self, frame_or_observation: Any, frame_skip: int | float = 0) -> np.ndarray:
        frame = self._frame_obs(frame_or_observation)
        if self._stack is None:
            return self.reset(frame, frame_skip=frame_skip)
        self._stack = step_frame_stack(
            self._stack,
            frame,
            base_channels=self.observation_config.base_channels,
        )
        self._last_observation = build_stacked_observation(
            self._stack,
            self.observation_config,
            frame_skip=frame_skip,
        )
        return self._last_observation.copy()

    def build_observation(
        self,
        frame_or_observation: Any,
        env_step: int | None = None,
        frame_skip: int | float | None = None,
    ) -> np.ndarray:
        frame, inferred_env_step, inferred_frame_skip = self._extract_inputs(
            frame_or_observation,
            env_step=env_step,
            frame_skip=frame_skip,
        )
        if inferred_env_step == 0 or self._stack is None:
            return self.reset(frame, frame_skip=inferred_frame_skip)
        return self.step(frame, frame_skip=inferred_frame_skip)

    def build_tensor(
        self,
        frame_or_observation: Any,
        env_step: int | None = None,
        frame_skip: int | float | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        obs = self.build_observation(
            frame_or_observation,
            env_step=env_step,
            frame_skip=frame_skip,
        )
        return hwc_to_nchw_tensor(obs, device=device)

    def _frame_obs(self, frame_or_observation: Any) -> np.ndarray:
        frame = _extract_frame(frame_or_observation)
        return resize_frame_pil(
            frame,
            image_size=self.observation_config.image_size,
            grayscale=self.observation_config.grayscale,
        )

    def _extract_inputs(
        self,
        frame_or_observation: Any,
        env_step: int | None,
        frame_skip: int | float | None,
    ) -> tuple[Any, int, int | float]:
        frame = _extract_frame(frame_or_observation)
        inferred_env_step = env_step
        inferred_frame_skip = frame_skip
        metadata = getattr(frame_or_observation, "metadata", None)
        if inferred_env_step is None and hasattr(frame_or_observation, "env_step"):
            inferred_env_step = int(getattr(frame_or_observation, "env_step"))
        if inferred_env_step is None:
            inferred_env_step = 0 if self._stack is None else 1
        if inferred_frame_skip is None and isinstance(metadata, Mapping):
            inferred_frame_skip = metadata.get("frame_skip", metadata.get("latency_frame_skip", 0))
        if inferred_frame_skip is None:
            inferred_frame_skip = 0
        return frame, int(inferred_env_step), inferred_frame_skip


def _extract_frame(frame_or_observation: Any) -> Any:
    if hasattr(frame_or_observation, "data"):
        return getattr(frame_or_observation, "data")
    return frame_or_observation
