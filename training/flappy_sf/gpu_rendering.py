from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict as GymDict

from modeling.flappy_ppo.observation import FlappyPpoObservationConfig
from modeling.flappy_ppo.preprocessing import FlappyPpoTorchObservationPipeline
from training.flappy_ppo.paths import ensure_flappy_gymnasium_on_path


ensure_flappy_gymnasium_on_path()

from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState  # noqa: E402
from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets, TorchFlappyRenderer  # noqa: E402


def resolve_gpu_render_device(device: str | torch.device) -> torch.device:
    requested = str(device).strip().lower()
    if requested == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Flappy GPU rendering requested gpu_render_device=auto, but CUDA is unavailable. "
                "Use gpu_render_device=cpu for CPU-compatible tests."
            )
        return torch.device("cuda")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Flappy GPU rendering requested gpu_render_device={device!s}, but CUDA is unavailable. "
                "Use gpu_render_device=cpu for CPU-compatible tests."
            )
        return torch.device(requested)
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("gpu_render_device must be 'auto', 'cpu', 'cuda', or a cuda device string")


class FlappyGpuRenderedObsWrapper(gym.Wrapper):
    """Render Flappy observations through Torch and expose Sample Factory's batched multi-agent contract."""

    num_agents = 1
    is_multiagent = True

    def __init__(
        self,
        env: gym.Env,
        *,
        obs_resize: tuple[int, int] | None,
        frame_stack: int,
        obs_mode: str,
        frame_skip_values: Sequence[int],
        gpu_render_device: str | torch.device,
    ):
        super().__init__(env)
        if obs_resize is None:
            raise ValueError("use_gpu_render=true requires env.obs_resize to be set")
        height, width = (int(obs_resize[0]), int(obs_resize[1]))
        if height != width:
            raise ValueError("use_gpu_render=true currently requires square env.obs_resize")

        self.render_device = resolve_gpu_render_device(gpu_render_device)
        self.observation_config = FlappyPpoObservationConfig(
            image_size=height,
            grayscale=False,
            stack_frames=int(frame_stack),
            obs_mode=str(obs_mode),
            frame_skip_values=tuple(int(item) for item in frame_skip_values),
        )
        self.observation_space = GymDict(
            {
                "obs": Box(
                    low=0,
                    high=255,
                    shape=(
                        self.observation_config.in_channels,
                        self.observation_config.image_size,
                        self.observation_config.image_size,
                    ),
                    dtype=np.uint8,
                )
            }
        )
        self.pipeline = FlappyPpoTorchObservationPipeline(
            self.observation_config,
            n_envs=1,
            device=self.render_device,
        )
        self.renderer: TorchFlappyRenderer | None = None
        self._raw_rgb_buffer: torch.Tensor | None = None

    def reset(self, **kwargs):
        numeric_obs, info = self.env.reset(**kwargs)
        info = self._with_numeric_obs(info, numeric_obs)
        obs = self.pipeline.reset(self._render_raw_rgb(), frame_skips=self._frame_skips(info))
        return self._obs(obs), [info]

    def step(self, action):
        scalar_action = self._scalar_action(action)
        numeric_obs, reward, terminated, truncated, info = self.env.step(scalar_action)
        info = self._with_numeric_obs(info, numeric_obs)
        done = bool(terminated or truncated)

        if done:
            reset_numeric_obs, reset_info = self.env.reset()
            reset_info = self._with_numeric_obs(reset_info, reset_numeric_obs)
            info["reset_info"] = reset_info
            obs = self.pipeline.reset(self._render_raw_rgb(), frame_skips=self._frame_skips(reset_info))
        else:
            obs = self.pipeline.step(
                self._render_raw_rgb(),
                dones=np.zeros(1, dtype=np.bool_),
                frame_skips=self._frame_skips(info),
            )

        return (
            self._obs(obs),
            np.asarray([float(reward)], dtype=np.float32),
            np.asarray([bool(terminated)], dtype=np.bool_),
            np.asarray([bool(truncated)], dtype=np.bool_),
            [info],
        )

    def _obs(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if obs.ndim != 4 or int(obs.shape[0]) != 1:
            raise RuntimeError(f"Expected batched observation tensor with shape (1,C,H,W), got {tuple(obs.shape)}")
        return {"obs": obs}

    def _render_raw_rgb(self) -> torch.Tensor:
        state = self.env.unwrapped.get_render_state()
        if self.renderer is None:
            assets = FlappyRenderAssets.for_state(state)
            self.renderer = TorchFlappyRenderer(assets, device=self.render_device)
        batch = BatchedFlappyRenderState.from_states([state])
        return self.renderer.render_into(batch, self._raw_rgb_output_buffer(batch))

    def _raw_rgb_output_buffer(self, batch: BatchedFlappyRenderState) -> torch.Tensor:
        expected_shape = (int(batch.batch_size), int(batch.screen_height), int(batch.screen_width), 3)
        expected_device = self.render_device if self.renderer is None else self.renderer.device
        if (
            self._raw_rgb_buffer is None
            or tuple(self._raw_rgb_buffer.shape) != expected_shape
            or self._raw_rgb_buffer.device != expected_device
        ):
            self._raw_rgb_buffer = torch.empty(expected_shape, dtype=torch.uint8, device=expected_device)
        return self._raw_rgb_buffer

    @staticmethod
    def _with_numeric_obs(info: dict[str, Any] | None, numeric_obs: Any) -> dict[str, Any]:
        out = dict(info or {})
        out["numeric_obs"] = numeric_obs
        return out

    @staticmethod
    def _frame_skips(info: dict[str, Any]) -> np.ndarray:
        frame_skip = info.get("frame_skip", info.get("latency_frame_skip", info.get("queue_latency", 0)))
        return np.asarray([int(frame_skip)], dtype=np.int64)

    @staticmethod
    def _scalar_action(action: Any) -> Any:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return int(action.reshape(-1)[0])
            return action
        if isinstance(action, (list, tuple)) and len(action) == 1:
            return FlappyGpuRenderedObsWrapper._scalar_action(action[0])
        return action
