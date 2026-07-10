from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict as GymDict

from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState
from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets, TorchFlappyRenderer
from latency_bench.envs.flappy import (
    FlappyObservationConfig,
    FlappyTorchObservationPipeline,
)
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY, RawRgbFrameStackBuffer


_PROFILE_FIELDS = (
    "normalize_actions_s",
    "env_step_s",
    "fused_decision_s",
    "latency_begin_s",
    "latency_raw_actions_s",
    "core_step_s",
    "latency_record_s",
    "latency_finish_s",
    "metrics_update_s",
    "auto_reset_s",
    "get_render_state_s",
    "state_batch_from_states_s",
    "render_into_s",
    "pipeline_s",
    "reset_obs_total_s",
    "step_obs_total_s",
)


class _GpuRenderProfiler:
    def __init__(
        self,
        *,
        enabled: bool,
        interval: int,
        device: torch.device,
        worker_index: int,
        vector_index: int,
        env_id: int,
    ):
        self.enabled = bool(enabled)
        self.interval = max(1, int(interval))
        self.device = torch.device(device)
        self.worker_index = int(worker_index)
        self.vector_index = int(vector_index)
        self.env_id = int(env_id)
        self.total_steps = 0
        self.totals = {field: 0.0 for field in _PROFILE_FIELDS}
        self.counts = {field: 0 for field in _PROFILE_FIELDS}

    def time(self, field: str):
        if not self.enabled:
            return nullcontext()
        return _GpuRenderProfileTimer(self, field)

    def record(self, field: str, elapsed: float) -> None:
        if field not in self.totals:
            self.totals[field] = 0.0
            self.counts[field] = 0
        self.totals[field] += float(elapsed)
        self.counts[field] += 1

    def synchronize(self) -> None:
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def observe_step(self) -> None:
        if not self.enabled:
            return
        self.total_steps += 1
        if self.total_steps % self.interval == 0:
            self.log_and_reset()

    def log_and_reset(self) -> None:
        parts = [
            "[FlappyGpuRenderProfile]",
            f"pid={os.getpid()}",
            f"worker={self.worker_index}",
            f"vector={self.vector_index}",
            f"env_id={self.env_id}",
            f"steps={self.total_steps}",
        ]
        for field in _PROFILE_FIELDS:
            count = self.counts.get(field, 0)
            avg_ms = 0.0 if count <= 0 else 1000.0 * self.totals[field] / float(count)
            parts.append(f"{field}_avg_ms={avg_ms:.3f}")
            parts.append(f"{field}_n={count}")
        print(" ".join(parts), flush=True)
        for field in _PROFILE_FIELDS:
            self.totals[field] = 0.0
            self.counts[field] = 0


class _GpuRenderProfileTimer:
    def __init__(self, profiler: _GpuRenderProfiler, field: str):
        self.profiler = profiler
        self.field = field
        self.start = 0.0

    def __enter__(self):
        self.profiler.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.profiler.synchronize()
        self.profiler.record(self.field, time.perf_counter() - self.start)
        return False


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
        gpu_render_device: str | torch.device,
        gpu_render_profile: bool = False,
        gpu_render_profile_interval: int = 1000,
        export_env_raw_rgb_frames: bool = False,
        worker_index: int = 0,
        vector_index: int = 0,
        env_id: int = 0,
    ):
        super().__init__(env)
        height, width = (int(obs_resize[0]), int(obs_resize[1]))
        if height != width:
            raise ValueError("use_gpu_render=true currently requires square env.obs_resize")

        self.render_device = resolve_gpu_render_device(gpu_render_device)
        self.export_env_raw_rgb_frames = bool(export_env_raw_rgb_frames)
        self.observation_config = FlappyObservationConfig(
            image_size=height,
            grayscale=False,
            stack_frames=int(frame_stack),
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
        self.pipeline = FlappyTorchObservationPipeline(
            self.observation_config,
            n_envs=1,
            device=self.render_device,
        )
        self.renderer: TorchFlappyRenderer | None = None
        self._raw_rgb_buffer: torch.Tensor | None = None
        self._last_raw_rgb: torch.Tensor | None = None
        self._env_raw_rgb_frames = RawRgbFrameStackBuffer(num_frames=int(frame_stack))
        self.gpu_render_profiler = _GpuRenderProfiler(
            enabled=bool(gpu_render_profile),
            interval=int(gpu_render_profile_interval),
            device=self.render_device,
            worker_index=int(worker_index),
            vector_index=int(vector_index),
            env_id=int(env_id),
        )

    def reset(self, **kwargs):
        numeric_obs, info = self.env.reset(**kwargs)
        info = self._with_numeric_obs(info, numeric_obs)
        with self.gpu_render_profiler.time("reset_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            self._last_raw_rgb = raw_rgb
            self._reset_env_raw_rgb_info(info, raw_rgb)
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.reset(raw_rgb)
        return self._obs(obs), [info]

    def step(self, action):
        scalar_action = self._scalar_action(action)
        with self.gpu_render_profiler.time("env_step_s"):
            numeric_obs, reward, terminated, truncated, info = self.env.step(scalar_action)
        info = self._with_numeric_obs(info, numeric_obs)
        done = bool(terminated or truncated)

        with self.gpu_render_profiler.time("step_obs_total_s"):
            if done:
                reset_numeric_obs, reset_info = self.env.reset()
                reset_info = self._with_numeric_obs(reset_info, reset_numeric_obs)
                info["reset_info"] = reset_info
                raw_rgb = self._render_raw_rgb()
                self._last_raw_rgb = raw_rgb
                self._reset_env_raw_rgb_info(info, raw_rgb)
                with self.gpu_render_profiler.time("pipeline_s"):
                    obs = self.pipeline.reset(raw_rgb)
            else:
                raw_rgb = self._render_raw_rgb()
                self._last_raw_rgb = raw_rgb
                self._append_env_raw_rgb_info(info, raw_rgb)
                with self.gpu_render_profiler.time("pipeline_s"):
                    obs = self.pipeline.step(
                        raw_rgb,
                        dones=np.zeros(1, dtype=np.bool_),
                    )
        self.gpu_render_profiler.observe_step()

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
        with self.gpu_render_profiler.time("get_render_state_s"):
            state = self.env.unwrapped.get_render_state()
        if self.renderer is None:
            assets = FlappyRenderAssets.for_state(state)
            self.renderer = TorchFlappyRenderer(assets, device=self.render_device)
        with self.gpu_render_profiler.time("state_batch_from_states_s"):
            batch = BatchedFlappyRenderState.from_states([state])
        with self.gpu_render_profiler.time("render_into_s"):
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

    def _reset_env_raw_rgb_info(self, info: dict[str, Any], raw_rgb: torch.Tensor) -> None:
        if self.export_env_raw_rgb_frames:
            self._env_raw_rgb_frames.reset(self._raw_rgb_frame(raw_rgb))
            info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._env_raw_rgb_frames.stack()

    def _append_env_raw_rgb_info(self, info: dict[str, Any], raw_rgb: torch.Tensor) -> None:
        if self.export_env_raw_rgb_frames:
            self._env_raw_rgb_frames.append(self._raw_rgb_frame(raw_rgb))
            info[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._env_raw_rgb_frames.stack()

    @staticmethod
    def _raw_rgb_frame(raw_rgb: torch.Tensor) -> np.ndarray:
        return raw_rgb.detach().cpu().numpy()[0]

    def render_game_frame(self) -> np.ndarray:
        """Return the unprocessed Flappy game RGB frame for video recording."""
        return self._raw_rgb_frame(self._last_raw_rgb)

    @staticmethod
    def _with_numeric_obs(info: dict[str, Any] | None, numeric_obs: Any) -> dict[str, Any]:
        out = dict(info or {})
        out["numeric_obs"] = numeric_obs
        return out

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
