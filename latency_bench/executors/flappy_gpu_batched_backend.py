from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore
from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState
from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets, TorchFlappyRenderer
from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.executors.env_step_backend import EnvStepResponse, RemoteEnvSlotHandle
from latency_bench.envs.flappy import (
    FlappyObservationConfig,
    FlappyTorchObservationPipeline,
)
from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY, RawRgbFrameStackBuffer
from training.flappy_sf.gpu_rendering import _GpuRenderProfiler, resolve_gpu_render_device
from training.flappy_sf.flappy_cuda_extension import CudaFlappyRenderer
from training.flappy_sf.gpu_core import CudaFlappyBirdCore


class FlappyGpuBatchedEnvStepBackend:
    """Batch Flappy physics and Torch rendering behind latency_bench's scheduler.

    This backend deliberately does not use training.flappy_sf.batched_gpu_env:
    latency_bench already owns action latency and persistence through one
    ActionScheduler per slot. The backend only applies the active action for
    one raw frame per slot and returns observations with the same contract as
    FlappyBirdEnvAdapter(use_gpu_render=True).
    """

    def __init__(self, *, config: dict[str, Any], noop_action: Action, num_slots: int):
        self.config = config
        self.noop_action = noop_action
        self._num_slots = int(num_slots)
        env_cfg = config["env"]
        self._env_fps = float(env_cfg["env_fps"])
        self._frame_ms = 1000.0 / self._env_fps
        self.frame_stack = int(env_cfg.get("frame_stack", 1) or 1)
        self.policy_observation_layout = self._policy_observation_layout(config)
        self._returns_torch_chw_observations = self.policy_observation_layout == "chw"
        self.export_env_raw_rgb_frames = self._uses_raw_rgb_vla_policy(config)
        self._updates_raw_rgb_on_step = self.export_env_raw_rgb_frames
        self.obs_resize = tuple(int(value) for value in env_cfg["obs_resize"])
        self.render_device = resolve_gpu_render_device(str(env_cfg.get("gpu_render_device", "auto")))

        screen_size = env_cfg.get("screen_size")
        core_kwargs: dict[str, Any] = {
            "batch_size": self._num_slots,
            "normalize_obs": bool(env_cfg.get("normalize_obs", True)),
            "pipe_gap": int(env_cfg.get("pipe_gap", 100)),
            "bird_color": env_cfg.get("bird_color", "yellow"),
            "pipe_color": env_cfg.get("pipe_color", "green"),
            "background": env_cfg.get("background", "day"),
            "score_limit": env_cfg.get("score_limit"),
        }
        if screen_size is not None:
            core_kwargs["screen_size"] = tuple(int(value) for value in screen_size)
        simulator = env_cfg["simulator"]
        core_cls = {
            "cpu": BatchedFlappyBirdCore,
            "gpu": CudaFlappyBirdCore,
        }[simulator]
        if core_cls is CudaFlappyBirdCore:
            core_kwargs["device"] = self.render_device
        self.core = core_cls(**core_kwargs)

        height = int(self.obs_resize[0])
        self.pipeline = FlappyTorchObservationPipeline(
            FlappyObservationConfig(
                image_size=height,
                grayscale=False,
                stack_frames=self.frame_stack if self._returns_torch_chw_observations else 1,
            ),
            n_envs=self._num_slots,
            device=self.render_device,
        )
        self.renderer: TorchFlappyRenderer | None = None
        self._raw_rgb_buffer: torch.Tensor | None = None
        self._pipeline_has_reset = False
        self._last_frames: list[Any] = [None for _slot in range(self._num_slots)]
        self._observed_frames: list[deque[np.ndarray]] = [
            deque(maxlen=self.frame_stack) for _slot in range(self._num_slots)
        ]
        self._env_raw_rgb_frames = [
            RawRgbFrameStackBuffer(num_frames=self.frame_stack)
            for _slot in range(self._num_slots)
        ]
        self._last_numeric_obs: list[Any] = [None for _slot in range(self._num_slots)]
        self._last_infos: list[dict[str, Any]] = [{} for _slot in range(self._num_slots)]
        self._env_steps = np.zeros(self._num_slots, dtype=np.int64)
        self._last_frame_skips = np.zeros(self._num_slots, dtype=np.int64)
        self.slot_handles = [
            RemoteEnvSlotHandle(
                slot_id=slot_id,
                env_fps=self._env_fps,
                noop_action=noop_action,
                action_space=gym.spaces.Discrete(2),
            )
            for slot_id in range(self._num_slots)
        ]
        self.last_reset_metadata_by_slot: dict[int, dict[str, Any]] = {}
        self.last_step_metadata_by_slot: dict[int, dict[str, Any]] = {}
        self.closed = False
        self.gpu_render_profiler = _GpuRenderProfiler(
            enabled=bool(env_cfg.get("gpu_render_profile", False)),
            interval=int(env_cfg.get("gpu_render_profile_interval", 1000) or 1000),
            device=self.render_device,
            worker_index=0,
            vector_index=0,
            env_id=0,
        )

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def env_fps(self) -> float:
        return self._env_fps

    def worker_pids(self) -> list[int]:
        return []

    def reset_slot(self, slot_id: int, *, episode_id: int, seed: int | None) -> Observation:
        self._raise_if_closed()
        slot_id = int(slot_id)
        start = time.perf_counter()
        numeric_obs, infos = self._reset_core_slots([slot_id], seeds=[seed])
        info = self._with_numeric_obs(infos[slot_id], numeric_obs[slot_id])
        frames = self._render_observation_frames_for_reset([slot_id])

        self._env_steps[slot_id] = 0
        self._observed_frames[slot_id].clear()
        self._last_frames[slot_id] = frames[slot_id]
        self._last_numeric_obs[slot_id] = numeric_obs[slot_id]
        self._last_infos[slot_id] = info
        observation = self._make_observation(slot_id, numeric_obs=numeric_obs[slot_id], info=info)
        metadata = {
            "backend": "flappy_gpu_batched",
            "slot_id": slot_id,
            "episode_id": int(episode_id),
            "worker_pid": os.getpid(),
            "worker_reset_wall_sec": time.perf_counter() - start,
        }
        self.last_reset_metadata_by_slot[slot_id] = metadata
        self.slot_handles[slot_id].update_observation(observation, metadata)
        return observation

    def observe_slots(self, slot_ids: Sequence[int]) -> dict[int, Observation]:
        self._raise_if_closed()
        slot_ids = [int(slot_id) for slot_id in slot_ids]
        frames = self._render_observation_frames_for_observe(slot_ids)
        observations: dict[int, Observation] = {}
        for slot_id in slot_ids:
            self._last_frames[slot_id] = frames[slot_id]
            observation = self._observe_slot(slot_id)
            self.slot_handles[slot_id].update_observation(observation, None)
            observations[slot_id] = observation
        return observations

    def step_slots(self, actions_by_slot: Mapping[int, Action]) -> dict[int, EnvStepResponse]:
        self._raise_if_closed()
        start = time.perf_counter()
        slot_ids = [int(slot_id) for slot_id in actions_by_slot]
        actions = np.zeros(self._num_slots, dtype=np.int64)
        active_mask = np.zeros(self._num_slots, dtype=np.bool_)
        action_names: dict[int, str | None] = {}
        for slot_id, action in actions_by_slot.items():
            slot_id = int(slot_id)
            actions[slot_id] = int(action.value)
            active_mask[slot_id] = True
            action_names[slot_id] = action.name

        with self.gpu_render_profiler.time("env_step_s"):
            step_result = self.core.step_batch(actions, active_mask=active_mask)

        terminal_infos = self._terminal_infos(step_result, slot_ids)
        done_mask = active_mask & np.logical_or(step_result.terminated, step_result.truncated)
        self._auto_reset_done_slots(done_mask, terminal_infos)
        if self._updates_raw_rgb_on_step:
            self._render_raw_rgb_for_step_metadata(done_mask=done_mask, active_mask=active_mask)
        self.gpu_render_profiler.observe_step()

        responses: dict[int, EnvStepResponse] = {}
        for slot_id in slot_ids:
            self._env_steps[slot_id] += 1
            info = terminal_infos[slot_id]
            numeric_obs = step_result.observations[slot_id]
            self._last_numeric_obs[slot_id] = numeric_obs
            self._last_infos[slot_id] = info

            step_info = dict(info)
            step_info.pop("numeric_obs", None)
            step_info.update(
                {
                    "env_step": int(self._env_steps[slot_id]),
                    "sim_time_ms": int(self._env_steps[slot_id]) * self._frame_ms,
                    "applied_action": int(actions[slot_id]),
                    "applied_action_name": action_names[slot_id],
                }
            )
            result = StepResult(
                observation=None,
                reward=float(np.float32(step_result.rewards[slot_id])),
                done=bool(step_result.terminated[slot_id]),
                truncated=bool(step_result.truncated[slot_id]),
                info=step_info,
            )
            metadata = {
                "backend": "flappy_gpu_batched",
                "slot_id": slot_id,
                "worker_pid": os.getpid(),
                "worker_step_wall_sec": time.perf_counter() - start,
            }
            self.last_step_metadata_by_slot[slot_id] = metadata
            responses[slot_id] = EnvStepResponse(
                result=result,
                worker_pid=os.getpid(),
                metadata=metadata,
            )
        return responses

    def close(self) -> None:
        self.closed = True

    def _render_observation_frames_for_reset(self, slot_ids: Sequence[int]) -> list[np.ndarray]:
        with self.gpu_render_profiler.time("reset_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            self._reset_env_raw_rgb_frames(raw_rgb, slot_ids)
            reset_mask = np.zeros(self._num_slots, dtype=np.bool_)
            for slot_id in slot_ids:
                reset_mask[int(slot_id)] = True
            with self.gpu_render_profiler.time("pipeline_s"):
                if self._pipeline_has_reset:
                    obs = self.pipeline.reset_indices(
                        raw_rgb,
                        reset_mask=reset_mask,
                        frame_skips=self._last_frame_skips,
                    )
                else:
                    obs = self.pipeline.reset(raw_rgb, frame_skips=self._last_frame_skips)
                    self._pipeline_has_reset = True
        return self._policy_observation_frames(obs)

    def _render_observation_frames_for_observe(self, slot_ids: Sequence[int]) -> list[np.ndarray]:
        with self.gpu_render_profiler.time("observe_obs_total_s"):
            raw_rgb = self._render_raw_rgb_for_observe(slot_ids)
            active_mask = np.zeros(self._num_slots, dtype=np.bool_)
            for slot_id in slot_ids:
                active_mask[int(slot_id)] = True
            if not self._updates_raw_rgb_on_step:
                self._update_env_raw_rgb_frames(
                    raw_rgb,
                    done_mask=np.zeros(self._num_slots, dtype=np.bool_),
                    active_mask=active_mask,
                )
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.step(
                    raw_rgb,
                    dones=np.zeros(self._num_slots, dtype=np.bool_),
                    frame_skips=self._last_frame_skips,
                    active_mask=active_mask,
                )
        return self._policy_observation_frames(obs)

    def _render_raw_rgb_for_step_metadata(
        self,
        *,
        done_mask: np.ndarray,
        active_mask: np.ndarray,
    ) -> None:
        with self.gpu_render_profiler.time("step_raw_rgb_total_s"):
            raw_rgb = self._render_raw_rgb()
            self._update_env_raw_rgb_frames(raw_rgb, done_mask=done_mask, active_mask=active_mask)

    def _render_raw_rgb_for_observe(self, slot_ids: Sequence[int]) -> torch.Tensor:
        del slot_ids
        return self._render_raw_rgb()

    def _render_raw_rgb(self) -> torch.Tensor:
        if isinstance(self.core, CudaFlappyBirdCore):
            if self.renderer is None:
                assets = FlappyRenderAssets.load(
                    bird_color=self.core.bird_color,
                    pipe_color=self.core.pipe_color,
                    background=self.core.background,
                )
                self.renderer = CudaFlappyRenderer(assets, device=self.render_device)
            with self.gpu_render_profiler.time("render_into_s"):
                return self.renderer.render_core_into(
                    self.core,
                    self._raw_rgb_output_buffer_for_shape(
                        batch_size=int(self.core.batch_size),
                        screen_height=int(self.core.screen_height),
                        screen_width=int(self.core.screen_width),
                    ),
                )

        with self.gpu_render_profiler.time("get_render_state_s"):
            batch = self.core.render_state_batch()
        if self.renderer is None:
            assets = FlappyRenderAssets.load(
                bird_color=batch.bird_color,
                pipe_color=batch.pipe_color,
                background=batch.background,
            )
            self.renderer = TorchFlappyRenderer(assets, device=self.render_device)
        with self.gpu_render_profiler.time("state_batch_from_states_s"):
            pass
        with self.gpu_render_profiler.time("render_into_s"):
            return self.renderer.render_into(batch, self._raw_rgb_output_buffer(batch))

    def _raw_rgb_output_buffer_for_shape(
        self,
        *,
        batch_size: int,
        screen_height: int,
        screen_width: int,
    ) -> torch.Tensor:
        expected_shape = (int(batch_size), int(screen_height), int(screen_width), 3)
        expected_device = self.render_device if self.renderer is None else self.renderer.device
        if (
            self._raw_rgb_buffer is None
            or tuple(self._raw_rgb_buffer.shape) != expected_shape
            or self._raw_rgb_buffer.device != expected_device
        ):
            self._raw_rgb_buffer = torch.empty(expected_shape, dtype=torch.uint8, device=expected_device)
        return self._raw_rgb_buffer

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

    def _terminal_infos(self, step_result: Any, slot_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        return {
            int(slot_id): self._with_numeric_obs(
                {"score": int(step_result.scores[int(slot_id)])},
                step_result.observations[int(slot_id)],
            )
            for slot_id in slot_ids
        }

    def _auto_reset_done_slots(self, done_mask: np.ndarray, terminal_infos: dict[int, dict[str, Any]]) -> None:
        done_slots = [int(slot_id) for slot_id in np.flatnonzero(done_mask)]
        if done_slots:
            reset_numeric_obs, reset_infos = self._reset_core_slots(
                done_slots,
                seeds=[None for _slot in done_slots],
            )
            for slot_id in done_slots:
                terminal_infos[slot_id]["reset_info"] = self._with_numeric_obs(
                    reset_infos[slot_id],
                    reset_numeric_obs[slot_id],
                )

    def _reset_core_slots(
        self,
        slot_ids: Sequence[int],
        *,
        seeds: Sequence[int | None],
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        indices = [int(slot_id) for slot_id in slot_ids]
        previous_ground_x = self.core.ground_x[indices]
        if isinstance(previous_ground_x, torch.Tensor):
            previous_ground_x = previous_ground_x.clone()
        else:
            previous_ground_x = previous_ground_x.copy()
        numeric_obs, infos = self.core.reset_batch(seeds=seeds, indices=indices)
        self.core.ground_x[indices] = previous_ground_x
        return numeric_obs, infos

    def _observe_slot(self, slot_id: int) -> Observation:
        metadata = self._observation_metadata(
            slot_id,
            numeric_obs=self._last_numeric_obs[slot_id],
            info=self._last_infos[slot_id],
        )
        if self._returns_torch_chw_observations:
            return Observation(
                data=self._last_frames[slot_id],
                env_step=int(self._env_steps[slot_id]),
                sim_time_ms=int(self._env_steps[slot_id]) * self._frame_ms,
                metadata=metadata,
            )
        frame = np.asarray(self._last_frames[slot_id], dtype=np.uint8)
        self._observed_frames[slot_id].append(frame)
        return Observation(
            data=self._policy_frame_stack(slot_id),
            env_step=int(self._env_steps[slot_id]),
            sim_time_ms=int(self._env_steps[slot_id]) * self._frame_ms,
            metadata=metadata,
        )

    def _make_observation(self, slot_id: int, *, numeric_obs: Any, info: dict[str, Any]) -> Observation:
        return Observation(
            data=self._reset_observation_data(slot_id),
            env_step=int(self._env_steps[slot_id]),
            sim_time_ms=int(self._env_steps[slot_id]) * self._frame_ms,
            metadata=self._observation_metadata(slot_id, numeric_obs=numeric_obs, info=info),
        )

    def _reset_observation_data(self, slot_id: int) -> Any:
        data = self._last_frames[slot_id]
        if self._returns_torch_chw_observations:
            return np.transpose(data[:3].detach().cpu().numpy(), (1, 2, 0)).copy()
        return data

    def _observation_metadata(self, slot_id: int, *, numeric_obs: Any, info: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(info)
        metadata.update(
            {
                "env_step": int(self._env_steps[slot_id]),
                "sim_time_ms": int(self._env_steps[slot_id]) * self._frame_ms,
                "numeric_obs": numeric_obs,
            }
        )
        if self.export_env_raw_rgb_frames:
            metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._env_raw_rgb_frames[int(slot_id)].stack()
        return metadata

    def _policy_frame_stack(self, slot_id: int) -> np.ndarray:
        frames = list(self._observed_frames[slot_id])
        if len(frames) < self.frame_stack:
            frames = [frames[0]] * (self.frame_stack - len(frames)) + frames
        frames = [np.asarray(frame, dtype=np.uint8) for frame in frames[-self.frame_stack :]]
        if self.policy_observation_layout == "chw":
            return np.concatenate([np.transpose(frame, (2, 0, 1)) for frame in frames], axis=0)
        if self.frame_stack == 1:
            return frames[-1]
        return np.concatenate(frames, axis=-1)

    @staticmethod
    def _policy_observation_layout(config: dict[str, Any]) -> str:
        policy_type = str(config.get("policy", {}).get("type", "")).strip().lower()
        return "chw" if policy_type == "flappy_sf" else "hwc"

    @staticmethod
    def _uses_raw_rgb_vla_policy(config: dict[str, Any]) -> bool:
        policy_type = str(config.get("policy", {}).get("type", "")).strip().lower()
        return policy_type in {"starvla", "flappy_openvla"}

    @staticmethod
    def _nchw_to_hwc_frames(obs: torch.Tensor) -> list[np.ndarray]:
        return [
            np.transpose(frame, (1, 2, 0)).copy()
            for frame in obs.detach().cpu().numpy()
        ]

    def _policy_observation_frames(self, obs: torch.Tensor) -> list[Any]:
        if self._returns_torch_chw_observations:
            return [obs[slot_id].detach() for slot_id in range(int(obs.shape[0]))]
        return self._nchw_to_hwc_frames(obs)

    def _reset_env_raw_rgb_frames(self, raw_rgb: torch.Tensor, slot_ids: Sequence[int]) -> None:
        if self.export_env_raw_rgb_frames:
            frames = raw_rgb.detach().cpu().numpy()
            for slot_id in slot_ids:
                slot_id = int(slot_id)
                self._env_raw_rgb_frames[slot_id].reset(frames[slot_id])

    def _update_env_raw_rgb_frames(
        self,
        raw_rgb: torch.Tensor,
        *,
        done_mask: np.ndarray,
        active_mask: np.ndarray,
    ) -> None:
        if self.export_env_raw_rgb_frames:
            frames = raw_rgb.detach().cpu().numpy()
            for slot_id in np.flatnonzero(active_mask):
                slot_id = int(slot_id)
                if bool(done_mask[slot_id]):
                    self._env_raw_rgb_frames[slot_id].reset(frames[slot_id])
                else:
                    self._env_raw_rgb_frames[slot_id].append(frames[slot_id])

    @staticmethod
    def _with_numeric_obs(info: dict[str, Any] | None, numeric_obs: Any) -> dict[str, Any]:
        out = dict(info or {})
        out["numeric_obs"] = numeric_obs
        return out

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("env step backend is closed")
