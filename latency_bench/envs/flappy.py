from __future__ import annotations

from collections import deque
from typing import Any

import flappy_bird_gymnasium  # noqa: F401
import gymnasium
import numpy as np
from PIL import Image

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter
from modeling.flappy_ppo.preprocessing import resize_frame_pil


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
        frame_stack: int = 1,
        use_gpu_render: bool = False,
        gpu_render_device: str = "auto",
        gpu_render_profile: bool = False,
        gpu_render_profile_interval: int = 1000,
        policy_observation_layout: str = "hwc",
    ):
        self.frame_stack = int(frame_stack)
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")

        self.use_gpu_render = bool(use_gpu_render)
        if self.use_gpu_render and obs_resize is None:
            raise ValueError("use_gpu_render=true requires obs_resize")

        self.gym_id = gym_id
        self.env_fps = float(env_fps)
        self.noop_action = noop_action or Action(value=0, name="noop", is_noop=True)
        self.obs_resize = tuple(obs_resize) if obs_resize is not None else None
        self.policy_observation_layout = str(policy_observation_layout).lower()
        self.env_step = 0
        self._last_info: dict[str, Any] = {}
        self._last_numeric_obs: Any = None
        self._last_frame: np.ndarray | None = None
        self._observed_frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

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
            )
        else:
            self.gym_env = base_env

        self.action_space = getattr(self.gym_env, "action_space", gymnasium.spaces.Discrete(2))
        if self.obs_resize is None:
            self.observation_space = getattr(self.gym_env, "observation_space", None)
        else:
            height, width = self.obs_resize
            self.observation_space = gymnasium.spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.env_fps

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        self._observed_frames.clear()

        if self.use_gpu_render:
            obs, infos = self.gym_env.reset(seed=seed)
            info = self._single_info(infos)
            numeric_obs = info.get("numeric_obs")
            frame = self._single_observation_hwc(obs)
        else:
            numeric_obs, info = self.gym_env.reset(seed=seed)
            info = dict(info or {})
            frame = self._render_rgb_frame()

        self._last_numeric_obs = numeric_obs
        self._last_info = info
        self._last_frame = frame
        return self._make_observation(numeric_obs=numeric_obs, info=info)

    def step(self, action: Action) -> StepResult:
        gym_action = int(action.value)

        if self.use_gpu_render:
            obs, reward, terminated, truncated, infos = self.gym_env.step(gym_action)
            info = self._single_info(infos)
            numeric_obs = info.get("numeric_obs")
            frame = self._single_observation_hwc(obs)
        else:
            numeric_obs, reward, terminated, truncated, info = self.gym_env.step(gym_action)
            info = dict(info or {})
            frame = self._render_rgb_frame()

        self.env_step += 1
        self._last_numeric_obs = numeric_obs
        self._last_info = info
        self._last_frame = frame

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
            done=bool(self._single_scalar(terminated)),
            truncated=bool(self._single_scalar(truncated)),
            info=step_info,
        )

    def observe(self) -> Observation:
        if self._last_frame is None:
            if self.use_gpu_render:
                raise RuntimeError("Flappy GPU render has no current frame; call reset() first")
            self._last_frame = self._render_rgb_frame()

        metadata = self._observation_metadata(numeric_obs=self._last_numeric_obs, info=self._last_info)
        frame = np.asarray(self._last_frame, dtype=np.uint8)
        self._observed_frames.append(frame)
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

    def _render_rgb_frame(self) -> np.ndarray:
        frame = self.gym_env.render()
        arr = self._rgb_hwc_frame(frame)
        if self.obs_resize is None:
            return arr
        height, width = self.obs_resize
        if height == width:
            arr = resize_frame_pil(arr, image_size=height, grayscale=False)
        else:
            image = Image.fromarray(arr).convert("RGB")
            image = image.resize((width, height), Image.BILINEAR)
            arr = np.asarray(image, dtype=np.uint8)
        if arr.shape[:2] != (height, width):
            raise RuntimeError(f"obs_resize expected HWC frame {(height, width, 3)}, got {arr.shape!r}")
        return self._rgb_hwc_frame(arr)

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

    def _observation_metadata(self, *, numeric_obs: Any, info: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(info)
        metadata.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "numeric_obs": numeric_obs,
            }
        )
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
        return cls._rgb_hwc_frame(arr)

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
