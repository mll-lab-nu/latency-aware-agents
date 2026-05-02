from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter
from modeling.flappy_ppo.preprocessing import resize_frame_pil


class FlappyBirdEnvAdapter(EnvAdapter):
    """Image-observation adapter for the local FlappyBird-v0 Gymnasium env."""

    def __init__(
        self,
        gym_id: str = "FlappyBird-v0",
        fps_env: float = 30.0,
        noop_action: Action | None = None,
        render_mode: str = "rgb_array",
        observation_mode: str = "image",
        use_lidar: bool = False,
        normalize_obs: bool = True,
        audio_on: bool = False,
        screen_size: tuple[int, int] | None = None,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        background: str | None = "day",
        score_limit: int | None = None,
        debug: bool = False,
        obs_resize: tuple[int, int] | None = None,
    ):
        if observation_mode != "image":
            raise ValueError("FlappyBirdEnvAdapter currently supports observation_mode='image' only")
        if render_mode != "rgb_array":
            raise ValueError("FlappyBirdEnvAdapter image mode requires render_mode='rgb_array'")
        if fps_env <= 0:
            raise ValueError("fps_env must be positive")

        gymnasium = _import_flappy_gymnasium()
        kwargs: dict[str, Any] = {
            "audio_on": bool(audio_on),
            "normalize_obs": bool(normalize_obs),
            "use_lidar": bool(use_lidar),
            "pipe_gap": int(pipe_gap),
            "bird_color": bird_color,
            "pipe_color": pipe_color,
            "render_mode": render_mode,
            "background": background,
            "score_limit": score_limit,
            "debug": bool(debug),
        }
        if screen_size is not None:
            kwargs["screen_size"] = tuple(screen_size)

        self.gym_env = gymnasium.make(gym_id, **kwargs)
        self.gym_id = gym_id
        self.fps_env = float(fps_env)
        self.noop_action = noop_action or Action(value=0, name="noop", is_noop=True)
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.obs_resize = obs_resize
        self.env_step = 0
        self._last_info: dict[str, Any] = {}
        self._last_numeric_obs: Any = None
        self._last_frame: Any = None

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.fps_env

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        numeric_obs, info = self.gym_env.reset(seed=seed)
        self._last_numeric_obs = numeric_obs
        self._last_info = dict(info or {})
        self._last_frame = self._render_rgb_frame()
        return self._make_observation(numeric_obs=numeric_obs, info=self._last_info)

    def step(self, action: Action) -> StepResult:
        gym_action = int(action.value)
        numeric_obs, reward, terminated, truncated, info = self.gym_env.step(gym_action)
        self.env_step += 1
        self._last_numeric_obs = numeric_obs
        self._last_info = dict(info or {})
        self._last_frame = self._render_rgb_frame()
        observation = self._make_observation(numeric_obs=numeric_obs, info=self._last_info)
        step_info = dict(self._last_info)
        step_info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "applied_action": gym_action,
                "applied_action_name": action.name,
                "observation_mode": self.observation_mode,
            }
        )
        return StepResult(
            observation=observation,
            reward=float(reward),
            done=bool(terminated),
            truncated=bool(truncated),
            info=step_info,
        )

    def render_frame(self) -> Any:
        if self._last_frame is None:
            self._last_frame = self._render_rgb_frame()
        return self._last_frame

    def get_state_info(self) -> dict[str, Any]:
        info = dict(self._last_info)
        info.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "score": info.get("score"),
                "observation_mode": self.observation_mode,
            }
        )
        return info

    def close(self) -> None:
        self.gym_env.close()

    def _render_rgb_frame(self) -> Any:
        frame = self.gym_env.render()
        if frame is None:
            raise RuntimeError("FlappyBird-v0 render() returned None; use render_mode='rgb_array'")
        if self.obs_resize is None:
            return frame
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError(f"Expected Flappy RGB frame with shape (H, W, 3), got {arr.shape!r}")
        height, width = self.obs_resize
        if height == width:
            arr = resize_frame_pil(arr, image_size=height, grayscale=False)
        else:
            image = Image.fromarray(arr).convert("RGB")
            image = image.resize((width, height), Image.BILINEAR)
            arr = np.asarray(image, dtype=np.uint8)
        if arr.shape[:2] != (height, width):
            raise RuntimeError(f"obs_resize expected HWC frame {(height, width, 3)}, got {arr.shape!r}")
        return np.transpose(arr, (2, 0, 1))

    def _make_observation(self, numeric_obs: Any, info: dict[str, Any]) -> Observation:
        metadata = dict(info)
        metadata.update(
            {
                "env_step": self.env_step,
                "sim_time_ms": self.env_step * self.frame_ms,
                "numeric_obs": numeric_obs,
                "observation_mode": self.observation_mode,
            }
        )
        return Observation(
            data=self._last_frame,
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )


def _import_flappy_gymnasium():
    try:
        import flappy_bird_gymnasium  # noqa: F401
        import gymnasium

        return gymnasium
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        flappy_root = repo_root / "flappy-bird-gymnasium"
        if flappy_root.exists() and str(flappy_root) not in sys.path:
            sys.path.insert(0, str(flappy_root))
        import flappy_bird_gymnasium  # noqa: F401
        import gymnasium

        return gymnasium
