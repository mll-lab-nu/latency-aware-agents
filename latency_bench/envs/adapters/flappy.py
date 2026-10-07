from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import flappy_bird_gymnasium  # noqa: F401
import gymnasium
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter, ENV_RAW_RGB_FRAME_STACK_INFO_KEY, RawRgbFrameStackBuffer


@dataclass(frozen=True)
class FlappyObservationConfig:
    image_size: int = 84
    grayscale: bool = False
    stack_frames: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", int(self.image_size))
        object.__setattr__(self, "grayscale", bool(self.grayscale))
        object.__setattr__(self, "stack_frames", int(self.stack_frames))

    @property
    def base_channels(self) -> int:
        return 1 if self.grayscale else 3

    @property
    def stacked_channels(self) -> int:
        return self.base_channels * self.stack_frames

    @property
    def in_channels(self) -> int:
        return self.stacked_channels


def resize_frame_method(frame: np.ndarray, size: int, method: str) -> np.ndarray:
    """Ablation helper: resize a native HWC uint8 frame to (size,size) with the
    chosen method. method in {bilinear_pil, inter_area_cv2, inter_linear_cv2, native}.
    'native' returns the frame unchanged. No silent fallback: cv2 methods raise if cv2 missing."""
    arr = np.asarray(frame, dtype=np.uint8)
    if method == "native":
        return arr
    side = int(size)
    if method == "bilinear_pil":
        img = Image.fromarray(arr).convert("RGB").resize((side, side), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    import cv2  # raises if unavailable — intentional, no fallback

    interp = {"inter_area_cv2": cv2.INTER_AREA, "inter_linear_cv2": cv2.INTER_LINEAR}
    if method not in interp:
        raise ValueError(f"unknown obs_resize_interp={method!r}")
    return cv2.resize(arr, (side, side), interpolation=interp[method]).astype(np.uint8)


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


def _resize_rgb_batch_torch_nchw(
    raw_rgb_batch: torch.Tensor,
    image_size: int,
    grayscale: bool = False,
) -> torch.Tensor:
    nchw = raw_rgb_batch.permute(0, 3, 1, 2).to(torch.float32)
    if grayscale:
        r = nchw[:, 0:1]
        g = nchw[:, 1:2]
        b = nchw[:, 2:3]
        nchw = torch.floor((r * 299.0 + g * 587.0 + b * 114.0 + 500.0) / 1000.0)

    resized = F.interpolate(
        nchw,
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (
        resized.round()
        .clamp_(0, 255)
        .to(torch.uint8)
        .contiguous()
    )


class FlappyTorchObservationPipeline:
    """Batch raw-RGB preprocessing pipeline for Flappy render observations."""

    def __init__(
        self,
        observation_config: FlappyObservationConfig,
        n_envs: int,
        device: str | torch.device,
    ) -> None:
        self.observation_config = observation_config
        self.n_envs = int(n_envs)
        self.device = torch.device(device)
        self.base_channels = observation_config.base_channels
        self._stack = torch.empty(
            (
                self.n_envs,
                observation_config.stacked_channels,
                observation_config.image_size,
                observation_config.image_size,
            ),
            dtype=torch.uint8,
            device=self.device,
        )
        self._next_stack = torch.empty_like(self._stack)

    def reset(
        self,
        raw_rgb_batch: torch.Tensor,
    ) -> torch.Tensor:
        frame_obs = self._resize(raw_rgb_batch)
        self._stack.copy_(frame_obs.repeat(1, self.observation_config.stack_frames, 1, 1))
        return self._stack

    def current(self) -> torch.Tensor:
        """Return the frame stack after the latest raw environment frame."""
        return self._stack

    def step(
        self,
        raw_rgb_batch: torch.Tensor,
        dones: np.ndarray,
        active_mask: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        frame_obs = self._resize(raw_rgb_batch)
        active = self._active_mask(active_mask)
        active_indices = np.flatnonzero(active)
        inactive_indices = np.flatnonzero(~active)
        done_indices = np.flatnonzero(np.asarray(dones, dtype=np.bool_) & active)
        if self.observation_config.stack_frames <= 1:
            if inactive_indices.size:
                self._next_stack[inactive_indices] = self._stack[inactive_indices]
                self._next_stack[active_indices] = frame_obs[active_indices]
            else:
                self._next_stack.copy_(frame_obs)
        elif inactive_indices.size:
            self._next_stack[inactive_indices] = self._stack[inactive_indices]
            self._next_stack[active_indices, : -self.base_channels, :, :] = self._stack[
                active_indices, self.base_channels :, :, :
            ]
            self._next_stack[active_indices, -self.base_channels :, :, :] = frame_obs[active_indices]
        else:
            self._next_stack[:, : -self.base_channels, :, :] = self._stack[:, self.base_channels :, :, :]
            self._next_stack[:, -self.base_channels :, :, :] = frame_obs
        if done_indices.size:
            self._next_stack[done_indices] = frame_obs[done_indices].repeat(
                1,
                self.observation_config.stack_frames,
                1,
                1,
            )
        self._stack, self._next_stack = self._next_stack, self._stack
        return self._stack

    def reset_indices(
        self,
        raw_rgb_batch: torch.Tensor,
        reset_mask: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        frame_obs = self._resize(raw_rgb_batch)
        mask = self._active_mask(reset_mask)
        reset_indices = [int(item) for item in np.flatnonzero(mask)]
        self._stack[reset_indices] = frame_obs[reset_indices].repeat(
            1,
            self.observation_config.stack_frames,
            1,
            1,
        )
        return self._stack

    def _active_mask(self, value: np.ndarray | torch.Tensor | None) -> np.ndarray:
        if value is None:
            return np.ones(self.n_envs, dtype=np.bool_)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.bool_).reshape(-1)

    def _resize(self, raw_rgb_batch: torch.Tensor) -> torch.Tensor:
        raw_rgb_batch = raw_rgb_batch.to(device=self.device, non_blocking=True)
        return _resize_rgb_batch_torch_nchw(
            raw_rgb_batch,
            image_size=self.observation_config.image_size,
            grayscale=self.observation_config.grayscale,
        )


class FlappyBirdEnvAdapter(EnvAdapter):
    """RGB image-observation adapter for the local FlappyBird-v0 Gymnasium env."""

    def __init__(
        self,
        gym_id: str = "FlappyBird-v0",
        env_fps: float = 30.0,
        noop_action: Action | None = None,
        screen_size: tuple[int, int] | None = None,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        background: str | None = "day",
        score_limit: int | None = None,
        obs_resize: tuple[int, int] | None = None,
        obs_resize_interp: str = "bilinear_pil",
        frame_stack: int = 1,
        use_gpu_render: bool = False,
        gpu_render_device: str = "auto",
        gpu_render_profile: bool = False,
        gpu_render_profile_interval: int = 1000,
        policy_observation_layout: str = "hwc",
        export_env_raw_rgb_frames: bool = False,
    ):
        self.frame_stack = frame_stack

        self.use_gpu_render = use_gpu_render
        if self.use_gpu_render and obs_resize is None:
            raise ValueError("use_gpu_render=true requires obs_resize")

        self.gym_id = gym_id
        self.env_fps = env_fps
        self.noop_action = noop_action or Action(value=0, name="noop", is_noop=True)
        self.obs_resize = tuple(obs_resize) if obs_resize is not None else None
        self.obs_resize_interp = obs_resize_interp
        self.policy_observation_layout = policy_observation_layout.lower()
        self.export_env_raw_rgb_frames = export_env_raw_rgb_frames
        self.env_step = 0
        self._last_info: dict[str, Any] = {}
        self._last_numeric_obs: Any = None
        self._last_frame: np.ndarray | None = None
        self._last_raw_frame: np.ndarray | None = None
        self._observed_frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._raw_rgb_frames = RawRgbFrameStackBuffer(num_frames=self.frame_stack)

        kwargs: dict[str, Any] = {
            "audio_on": False,
            "normalize_obs": True,
            "use_lidar": False,
            "pipe_gap": int(pipe_gap),
            "bird_color": bird_color,
            "pipe_color": pipe_color,
            "render_mode": None if self.use_gpu_render else "rgb_array",
            "background": background,
            "score_limit": score_limit,
            "debug": False,
        }
        if screen_size is not None:
            kwargs["screen_size"] = tuple(screen_size)

        base_env = gymnasium.make(gym_id, **kwargs)
        if self.use_gpu_render:
            from training.flappy_sf.gpu_rendering import FlappyGpuRenderedObsWrapper

            self.gym_env = FlappyGpuRenderedObsWrapper(
                base_env,
                obs_resize=self.obs_resize,
                frame_stack=1,
                gpu_render_device=gpu_render_device,
                gpu_render_profile=bool(gpu_render_profile),
                gpu_render_profile_interval=int(gpu_render_profile_interval),
                export_env_raw_rgb_frames=self.export_env_raw_rgb_frames,
            )
        else:
            self.gym_env = base_env

        self.action_space = getattr(self.gym_env, "action_space", gymnasium.spaces.Discrete(2))
        if self.obs_resize is None:
            self.observation_space = getattr(self.gym_env, "observation_space", None)
        else:
            height, width = self.obs_resize
            self.observation_space = gymnasium.spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        self._observed_frames.clear()

        if self.use_gpu_render:
            obs, infos = self.gym_env.reset(seed=seed)
            info = self._single_info(infos)
            numeric_obs = info.get("numeric_obs")
            frame = self._single_observation_hwc(obs)
            raw_frame = self._raw_rgb_frame_from_info(info) if self.export_env_raw_rgb_frames else frame
        else:
            numeric_obs, info = self.gym_env.reset(seed=seed)
            info = dict(info or {})
            raw_frame = self._rgb_hwc_frame(self.gym_env.render())
            frame = self._observation_frame(raw_frame)

        self._last_numeric_obs = numeric_obs
        self._last_info = info
        self._last_frame = frame
        if not self.use_gpu_render:
            self._last_raw_frame = self._rgb_hwc_frame(raw_frame)
        if not self.use_gpu_render or self.policy_observation_layout == "chw":
            self._reset_frame_stack(frame)
        if self.export_env_raw_rgb_frames:
            self._raw_rgb_frames.reset(raw_frame)
        return self._make_observation(numeric_obs=numeric_obs, info=info)

    def step(self, action: Action) -> StepResult:
        gym_action = int(action.value)

        if self.use_gpu_render:
            obs, reward, terminated, truncated, infos = self.gym_env.step(gym_action)
            info = self._single_info(infos)
            numeric_obs = info.get("numeric_obs")
            frame = self._single_observation_hwc(obs)
            raw_frame = self._raw_rgb_frame_from_info(info) if self.export_env_raw_rgb_frames else frame
        else:
            numeric_obs, reward, terminated, truncated, info = self.gym_env.step(gym_action)
            info = dict(info or {})
            raw_frame = self._rgb_hwc_frame(self.gym_env.render())
            frame = self._observation_frame(raw_frame)

        self.env_step += 1
        terminated_bool = bool(self._single_scalar(terminated))
        truncated_bool = bool(self._single_scalar(truncated))
        self._last_numeric_obs = numeric_obs
        self._last_info = info
        self._last_frame = frame
        if not self.use_gpu_render:
            self._last_raw_frame = self._rgb_hwc_frame(raw_frame)
        if not self.use_gpu_render or self.policy_observation_layout == "chw":
            self._append_observed_frame(frame)
        if self.export_env_raw_rgb_frames and not bool(terminated_bool or truncated_bool):
            self._raw_rgb_frames.append(raw_frame)

        observation = self._make_observation(numeric_obs=numeric_obs, info=info)
        step_info = dict(info)
        step_info.pop("numeric_obs", None)
        step_info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "applied_action": gym_action,
                "applied_action_name": action.name,
            }
        )
        return StepResult(
            observation=observation,
            reward=float(self._single_scalar(reward)),
            done=terminated_bool,
            truncated=truncated_bool,
            info=step_info,
        )

    def observe(self) -> Observation:
        if self._last_frame is None:
            if self.use_gpu_render:
                raise RuntimeError("Flappy GPU render has no current frame; call reset() first")
            self._last_frame = self._render_rgb_frame()

        metadata = self._observation_metadata(numeric_obs=self._last_numeric_obs, info=self._last_info)
        if self.use_gpu_render and self.policy_observation_layout != "chw":
            self._append_observed_frame(self._last_frame)
        return Observation(
            data=self._policy_frame_stack(),
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )

    def render_frame(self) -> Any:
        if self._last_frame is None:
            if self.use_gpu_render:
                raise RuntimeError("Flappy GPU render has no current frame; call reset() first")
            self._last_frame = self._render_rgb_frame()
        return self._last_frame

    def render_game_frame(self) -> np.ndarray:
        """Return the unprocessed Flappy game RGB frame for video recording."""
        if self.use_gpu_render:
            return self.gym_env.render_game_frame()
        return self._rgb_hwc_frame(self._last_raw_frame)

    def get_state_info(self) -> dict[str, Any]:
        info = dict(self._last_info)
        info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "score": info.get("score"),
            }
        )
        return info

    def close(self) -> None:
        self.gym_env.close()

    def _observation_frame(self, raw_frame: Any) -> np.ndarray:
        arr = self._rgb_hwc_frame(raw_frame)
        if self.obs_resize is None or self.obs_resize_interp == "native":
            return arr
        height, width = self.obs_resize
        if height == width and self.obs_resize_interp != "bilinear_pil":
            return self._rgb_hwc_frame(resize_frame_method(arr, height, self.obs_resize_interp))
        if height == width:
            arr = resize_frame_pil(arr, image_size=height, grayscale=False)
        else:
            image = Image.fromarray(arr).convert("RGB")
            image = image.resize((width, height), Image.BILINEAR)
            arr = np.asarray(image, dtype=np.uint8)
        if arr.shape[:2] != (height, width):
            raise RuntimeError(f"obs_resize expected HWC frame {(height, width, 3)}, got {arr.shape!r}")
        return self._rgb_hwc_frame(arr)

    def _render_rgb_frame(self) -> np.ndarray:
        return self._observation_frame(self.render_game_frame())

    @staticmethod
    def _rgb_hwc_frame(frame: Any) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3:
            raise RuntimeError(f"Expected Flappy RGB frame with 3 dimensions, got {arr.shape!r}")
        if arr.shape[-1] == 3:
            return arr
        if arr.shape[0] == 3:
            return np.transpose(arr, (1, 2, 0))
        raise RuntimeError(f"Expected Flappy RGB frame with shape (H, W, 3), got {arr.shape!r}")

    def _make_observation(self, numeric_obs: Any, info: dict[str, Any]) -> Observation:
        metadata = self._observation_metadata(numeric_obs=numeric_obs, info=info)
        return Observation(
            data=self._last_frame,
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )

    def _reset_frame_stack(self, frame: np.ndarray) -> None:
        self._observed_frames.clear()
        self._append_observed_frame(frame)

    def _append_observed_frame(self, frame: np.ndarray) -> None:
        self._observed_frames.append(np.asarray(frame, dtype=np.uint8))

    def _observation_metadata(self, *, numeric_obs: Any, info: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(info)
        metadata.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "numeric_obs": numeric_obs,
            }
        )
        if self.export_env_raw_rgb_frames:
            metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._raw_rgb_frames.stack()
        return metadata

    def _policy_frame_stack(self) -> np.ndarray:
        frames = list(self._observed_frames)
        if not frames:
            raise RuntimeError("Flappy observe() has no current frame; call reset() first")
        if len(frames) < self.frame_stack:
            frames = [frames[0]] * (self.frame_stack - len(frames)) + frames
        frames = [np.asarray(frame, dtype=np.uint8) for frame in frames[-self.frame_stack :]]
        if self.policy_observation_layout == "chw":
            chw_frames = [np.transpose(frame, (2, 0, 1)) for frame in frames]
            return np.concatenate(chw_frames, axis=0)
        if self.frame_stack == 1:
            return frames[-1]
        return np.concatenate(frames, axis=-1)

    @classmethod
    def _single_observation_hwc(cls, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            obs = obs["obs"]
        if hasattr(obs, "detach") and callable(obs.detach):
            obs = obs.detach().cpu().numpy()
        elif hasattr(obs, "cpu") and callable(obs.cpu):
            obs = obs.cpu().numpy()
        arr = np.asarray(obs)
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected one Flappy GPU env observation, got batch shape {arr.shape!r}")
            arr = arr[0]
        return cls._rgb_hwc_frame(arr).copy()

    @staticmethod
    def _raw_rgb_frame_from_info(info: dict[str, Any]) -> np.ndarray:
        return np.asarray(info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY])[-1]

    @staticmethod
    def _single_info(infos: Any) -> dict[str, Any]:
        if isinstance(infos, (list, tuple)):
            if len(infos) != 1:
                raise ValueError(f"Expected one Flappy GPU env info dict, got {len(infos)}")
            return dict(infos[0] or {})
        return dict(infos or {})

    @staticmethod
    def _single_scalar(value: Any) -> float:
        if hasattr(value, "detach") and callable(value.detach):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and callable(value.cpu):
            value = value.cpu().numpy()
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr.item())
        if arr.size != 1:
            raise ValueError(f"Expected scalar Flappy env value, got shape {arr.shape!r}")
        return float(arr.reshape(-1)[0])
