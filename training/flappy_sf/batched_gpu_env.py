from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict as GymDict, Discrete

import flappy_bird_gymnasium  # noqa: F401
from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore
from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState
from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets, TorchFlappyRenderer
from latency_bench.core.clock import EnvClock
from latency_bench.core.decision_action_history import DecisionActionHistory
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
from latency_bench.envs.adapters.flappy import (
    FlappyObservationConfig,
    FlappyTorchObservationPipeline,
)
from training.common.latency_config import (
    build_profile_sampler,
    profile_sampler_max_latency_raw_frames_for_workers,
    sample_factory_latency_params_from_flat_config,
    sample_factory_latency_seed_from_flat_config,
)
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY

from training.flappy_sf.gpu_rendering import _GpuRenderProfiler, resolve_gpu_render_device

from training.flappy_sf.batched_latency import BatchedFlappyLatencyScheduler
from training.common.batched_metrics import BatchedEpisodeMetrics
from training.flappy_sf.flappy_cuda_extension import CudaFlappyRenderer
from training.common.gpu_latency import CudaLatencyScheduler
from training.flappy_sf.gpu_core import CudaFlappyBirdCore


class FlappyBatchedGpuRenderedEnv(gym.Env):
    """A Sample Factory multi-agent env that batches Flappy Torch rendering per worker."""

    is_multiagent = True

    def __init__(
        self,
        *,
        cfg: Any,
        env_config: Any | None,
        obs_resize: tuple[int, int] | None,
        frame_stack: int,
        gpu_render_device: str | torch.device,
        gpu_render_batch_size: int,
        gpu_render_profile: bool = False,
        gpu_render_profile_interval: int = 1000,
    ):
        super().__init__()
        if obs_resize is None:
            raise ValueError("use_gpu_render=true requires env.obs_resize to be set")
        height, width = int(obs_resize[0]), int(obs_resize[1])
        if height != width:
            raise ValueError("use_gpu_render=true currently requires square env.obs_resize")

        self.cfg = cfg
        self.env_config = env_config
        self.num_agents = int(gpu_render_batch_size)
        if self.num_agents <= 0:
            raise ValueError("gpu_render_batch_size must be a positive integer")
        env_fps = getattr(cfg, "env_fps", None)
        obs_fps = getattr(cfg, "obs_fps", None)
        if env_fps is None or obs_fps is None:
            raise ValueError("Flappy Sample Factory GPU rendering requires env_fps and obs_fps")
        self.clock = EnvClock(env_fps=float(env_fps), obs_fps=float(obs_fps))

        self.render_device = resolve_gpu_render_device(gpu_render_device)
        self.observation_config = FlappyObservationConfig(
            image_size=height,
            grayscale=False,
            stack_frames=int(frame_stack),
        )
        self.export_env_raw_rgb_frames = bool(getattr(cfg, "export_env_raw_rgb_frames", False))
        self._env_raw_rgb_frames: list[deque[np.ndarray]] = [
            deque(maxlen=int(frame_stack)) for _ in range(self.num_agents)
        ]
        self.action_space = Discrete(2)
        history_decisions = getattr(cfg, "action_history_decisions", None)
        self.decision_action_history = (
            DecisionActionHistory(self.action_space, num_envs=self.num_agents, decisions=history_decisions)
            if history_decisions is not None else None
        )
        observation_spaces = {
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
        if self.decision_action_history is not None:
            observation_spaces["transport"] = Box(
                low=0, high=1, shape=(self.decision_action_history.observation_dim,), dtype=np.float32
            )
        self.observation_space = GymDict(observation_spaces)

        self.uses_batched_core = True
        score_limit = getattr(self.cfg, "score_limit", None)
        resolved_score_limit = None if score_limit in (None, "", 0, -1) else int(score_limit)
        simulator = self.cfg.simulator
        core_cls = {
            "cpu": BatchedFlappyBirdCore,
            "gpu": CudaFlappyBirdCore,
        }[simulator]
        core_kwargs = {}
        if core_cls is CudaFlappyBirdCore:
            core_kwargs["device"] = self.render_device
        self.core = core_cls(
            batch_size=self.num_agents,
            normalize_obs=bool(getattr(self.cfg, "normalize_obs", True)),
            pipe_gap=int(getattr(self.cfg, "pipe_gap", 100)),
            bird_color=getattr(self.cfg, "bird_color", "yellow"),
            pipe_color=getattr(self.cfg, "pipe_color", "green"),
            background=getattr(self.cfg, "background", "day"),
            score_limit=resolved_score_limit,
            **core_kwargs,
        )
        sub_seeds = [self._sub_seed(agent_idx) for agent_idx in range(self.num_agents)]
        latency_seeds = [
            sample_factory_latency_seed_from_flat_config(
                self.cfg,
                fallback_seed=sub_seeds[agent_idx],
                env_id=self._sub_env_id(agent_idx),
            )
            for agent_idx in range(self.num_agents)
        ]
        latency_params = sample_factory_latency_params_from_flat_config(self.cfg, env_fps=self.clock.env_fps)
        profile_samplers = None
        profile_max_latency_raw_frames = None
        if latency_params.profile_path is not None:
            profile_samplers = [
                build_profile_sampler(
                    latency_params.profile_path,
                    method=latency_params.profile_method,
                    seed=latency_seeds[agent_idx],
                    env_fps=self.clock.env_fps,
                    obs_fps=self.clock.obs_fps,
                )
                for agent_idx in range(self.num_agents)
            ]
            profile_max_latency_raw_frames = profile_sampler_max_latency_raw_frames_for_workers(
                profile_samplers[0],
                profile_worker_slot=latency_params.profile_worker_slot,
                simulated_worker_capacity=latency_params.simulated_worker_capacity,
                env_fps=self.clock.env_fps,
            )
        max_pending_actions = getattr(self.cfg, "max_pending_actions", None)
        latency_kwargs = {
            "batch_size": self.num_agents,
            "fixed_latency_raw_frames": latency_params.fixed_latency_raw_frames,
            "env_fps": self.clock.env_fps,
            "obs_fps": self.clock.obs_fps,
            "hold_policy": getattr(self.cfg, "hold_policy", ActionHoldPolicy.ONE_FRAME_THEN_NOOP.value),
            "ordering_policy": getattr(self.cfg, "ordering_policy", OrderingPolicy.LATEST_READY.value),
            "seeds": latency_seeds,
            "env_reset_seed_bases": [seed + 104729 for seed in sub_seeds],
            "add_latency_info": bool(getattr(self.cfg, "add_latency_info", False)),
            "profile_seed_by_reset": bool(getattr(self.cfg, "rollout_profile_seed_by_reset", False)),
            "simulated_worker_capacity": latency_params.simulated_worker_capacity,
        }
        if isinstance(self.core, CudaFlappyBirdCore):
            self.latency = CudaLatencyScheduler(
                **latency_kwargs,
                latency_distribution_spec=latency_params.latency_distribution_spec,
                sample_distribution_per_episode=latency_params.sample_distribution_per_episode,
                max_pending_actions=max_pending_actions,
                latency_samplers=profile_samplers,
                profile_worker_slot=latency_params.profile_worker_slot,
                profile_max_latency_raw_frames=profile_max_latency_raw_frames,
                device=self.render_device,
            )
        else:
            self.latency = BatchedFlappyLatencyScheduler(
                **latency_kwargs,
                latency_distribution_spec=latency_params.latency_distribution_spec,
                profile_path=latency_params.profile_path,
                profile_worker_slot=latency_params.profile_worker_slot,
                profile_method=latency_params.profile_method,
                sample_distribution_per_episode=latency_params.sample_distribution_per_episode,
                max_pending_actions=max_pending_actions,
            )
        self._uses_cuda_latency = isinstance(self.latency, CudaLatencyScheduler)
        self.metrics = BatchedEpisodeMetrics(
            batch_size=self.num_agents,
            frames_per_decision=self.clock.obs_stride_raw_frames,
            metrics_path=str(getattr(self.cfg, "episode_metrics_path", "") or ""),
            worker_index=self._worker_index,
            vector_index=self._vector_index,
            env_ids=[self._sub_env_id(agent_idx) for agent_idx in range(self.num_agents)],
            issued_action_labels={0: "noop", 1: "flap"},
        )
        self._raw_episode_steps = np.zeros(self.num_agents, dtype=np.int64)
        self._raw_episode_steps_gpu = torch.zeros(self.num_agents, dtype=torch.int64, device=self.render_device)
        self._batched_max_episode_steps = int(getattr(self.cfg, "max_episode_steps", 0) or 0)
        self.pipeline = FlappyTorchObservationPipeline(
            self.observation_config,
            n_envs=self.num_agents,
            device=self.render_device,
        )
        self.renderer: TorchFlappyRenderer | None = None
        self._raw_rgb_buffer: torch.Tensor | None = None
        self._has_reset = False
        self._active_agents = np.ones(self.num_agents, dtype=np.bool_)
        self._auto_reset_done = True
        self._last_observation_infos: list[dict[str, Any]] = [{} for _ in range(self.num_agents)]
        self._empty_info: dict[str, Any] = {}
        self.sample_factory_step_cpu_tensors: dict[str, torch.Tensor] = {}

        self.gpu_render_profiler = _GpuRenderProfiler(
            enabled=bool(gpu_render_profile),
            interval=int(gpu_render_profile_interval),
            device=self.render_device,
            worker_index=self._worker_index,
            vector_index=self._vector_index,
            env_id=self._env_id,
        )

    @property
    def _worker_index(self) -> int:
        return int(getattr(self.env_config, "worker_index", 0) or 0)

    @property
    def _vector_index(self) -> int:
        return int(getattr(self.env_config, "vector_index", 0) or 0)

    @property
    def _env_id(self) -> int:
        return int(getattr(self.env_config, "env_id", 0) or 0)

    def _sub_seed(self, agent_idx: int) -> int:
        base_seed = int(getattr(self.cfg, "seed", 0) or 0)
        # One Sample Factory env now contains all Flappy actors for a worker; agent_idx is the actor discriminator.
        return base_seed + 100000 * self._worker_index + 1000 * self._vector_index + int(agent_idx)

    def _sub_env_id(self, agent_idx: int) -> int:
        return self._env_id * self.num_agents + int(agent_idx)

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        del kwargs
        if seed is None:
            reset_seeds, latency_infos = self.latency.reset()
        else:
            reset_seeds, latency_infos = self.latency.reset(
                seeds=[int(seed) + int(agent_idx) for agent_idx in range(self.num_agents)]
            )
        self.metrics.reset()
        self._raw_episode_steps[:] = 0
        self._raw_episode_steps_gpu.zero_()
        if self.decision_action_history is not None:
            self.decision_action_history.reset()
        numeric_obs, core_infos = self.core.reset_batch(seeds=reset_seeds)
        infos = [
            self._with_numeric_obs({**core_infos[agent_idx], **latency_infos[agent_idx]}, numeric_obs[agent_idx])
            for agent_idx in range(self.num_agents)
        ]
        with self.gpu_render_profiler.time("reset_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            self._reset_env_raw_rgb_frames(raw_rgb)
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.reset(raw_rgb)
        self._has_reset = True
        self._attach_env_raw_rgb_infos(infos)
        self._last_observation_infos = [dict(info) for info in infos]
        return self._obs(obs), infos

    def set_auto_reset_done(self, enabled: bool) -> None:
        self._auto_reset_done = bool(enabled)

    def set_active_agents(self, active_mask: Sequence[bool]) -> None:
        mask = np.asarray(list(active_mask), dtype=np.bool_).reshape(-1)
        if int(mask.shape[0]) != self.num_agents:
            raise ValueError(f"Expected active mask with {self.num_agents} entries, got {int(mask.shape[0])}")
        self._active_agents = mask

    def reset_agents(
        self,
        reset_plan: Mapping[int, int],
        *,
        active_mask: Sequence[bool] | None = None,
    ):
        if active_mask is not None:
            self.set_active_agents(active_mask)
        if not self._has_reset:
            return self.reset()

        infos = [dict(info) for info in self._last_observation_infos]
        reset_mask = np.zeros(self.num_agents, dtype=np.bool_)
        with self.gpu_render_profiler.time("env_step_s"):
            reset_indices = [int(raw_agent_idx) for raw_agent_idx in reset_plan.keys()]
            reset_seeds_input = [int(raw_seed) for raw_seed in reset_plan.values()]
            reset_seeds, latency_infos = self.latency.reset(indices=reset_indices, seeds=reset_seeds_input)
            self.metrics.reset(indices=reset_indices)
            numeric_obs, core_infos = self.core.reset_batch(seeds=reset_seeds, indices=reset_indices)
            for offset, agent_idx in enumerate(reset_indices):
                info = self._with_numeric_obs(
                    {**core_infos[agent_idx], **latency_infos[offset]},
                    numeric_obs[agent_idx],
                )
                infos[agent_idx] = info
                self._last_observation_infos[agent_idx] = dict(info)
                reset_mask[agent_idx] = True
                self._raw_episode_steps[agent_idx] = 0
                self._raw_episode_steps_gpu[agent_idx] = 0
            if self.decision_action_history is not None:
                self.decision_action_history.reset(reset_indices)

        with self.gpu_render_profiler.time("reset_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            self._reset_env_raw_rgb_frames(raw_rgb, agent_indices=reset_indices)
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.reset_indices(
                    raw_rgb,
                    reset_mask=reset_mask,
                )
        self._attach_env_raw_rgb_infos(infos)
        return self._obs(obs), infos

    def step(self, action):
        if not self._has_reset:
            self.reset()

        with self.gpu_render_profiler.time("normalize_actions_s"):
            if self._uses_cuda_latency:
                actions = self._normalize_cuda_actions(action)
                issued_actions = actions.detach().cpu().numpy().astype(np.int64, copy=False)
            else:
                actions = np.asarray(self._normalize_actions(action), dtype=np.int64)
                issued_actions = actions
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminated = np.zeros(self.num_agents, dtype=np.bool_)
        truncated = np.zeros(self.num_agents, dtype=np.bool_)
        rewards_out: np.ndarray | torch.Tensor = rewards
        terminated_out: np.ndarray | torch.Tensor = terminated
        truncated_out: np.ndarray | torch.Tensor = truncated
        infos: list[dict[str, Any]] = [self._empty_info] * self.num_agents

        with self.gpu_render_profiler.time("env_step_s"):
            active = self._active_agents.copy()
            use_cuda_latency = self._uses_cuda_latency
            for agent_idx in np.flatnonzero(~active):
                observation_info = dict(self._last_observation_infos[agent_idx])
                observation_info["inactive"] = True
                infos[agent_idx] = observation_info

            if use_cuda_latency:
                active_for_latency = torch.as_tensor(active, dtype=torch.bool, device=self.render_device)
                with self.gpu_render_profiler.time("fused_decision_s"):
                    fused_result = self.latency.step_core_decision_numpy(
                        self.core,
                        actions,
                        active_mask=active_for_latency,
                        episode_step_idx=self._raw_episode_steps_gpu,
                        max_episode_steps=self._batched_max_episode_steps,
                    )
                rewards = fused_result.rewards.astype(np.float32, copy=False)
                terminated = fused_result.terminated
                truncated = fused_result.truncated
                rewards_out = fused_result.rewards_gpu
                terminated_out = fused_result.terminated_gpu
                truncated_out = fused_result.truncated_gpu
                raw_frames_advanced = fused_result.raw_frames_advanced
                self._record_decision_history(
                    fused_result.command_admitted, issued_actions, fused_result.applied_actions, active
                )
                self._raw_episode_steps[active] += raw_frames_advanced[active]
                timeout_mask = fused_result.timeout_mask
                latest_scores = fused_result.scores
                latency_raw_frames = fused_result.latency_raw_frames
                with self.gpu_render_profiler.time("metrics_update_s"):
                    done_infos = self.metrics.update(
                        issued_actions=issued_actions,
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        scores=latest_scores,
                        active_mask=active,
                        latency_raw_frames=latency_raw_frames,
                        command_admitted=fused_result.command_admitted,
                        timeout_mask=timeout_mask,
                    )
                done_mask = active & np.logical_or(terminated, truncated)
                if self.latency.add_latency_info:
                    latest_numeric_obs = self.core.observations()
                    for agent_idx in np.flatnonzero(active):
                        done = bool(done_mask[agent_idx])
                        latency_info = self.latency.core_decision_info_for_agent(int(agent_idx), fused_result)
                        info = {
                            "score": int(latest_scores[agent_idx]),
                            **latency_info,
                            "raw_reward": float(rewards[agent_idx]),
                        }
                        if bool(timeout_mask[agent_idx]):
                            info["TimeLimit.truncated"] = True
                        if int(agent_idx) in done_infos:
                            info.update(done_infos[int(agent_idx)])
                        info = self._with_numeric_obs(info, latest_numeric_obs[agent_idx])
                        if not (done and self._auto_reset_done):
                            self._last_observation_infos[agent_idx] = dict(info)
                        infos[agent_idx] = info
                    if self._auto_reset_done:
                        with self.gpu_render_profiler.time("auto_reset_s"):
                            reset_infos = self._auto_reset_done_agents(np.flatnonzero(done_mask))
                        for agent_idx, reset_info in reset_infos.items():
                            infos[agent_idx]["reset_info"] = reset_info
                else:
                    with self.gpu_render_profiler.time("auto_reset_s"):
                        reset_infos = (
                            self._auto_reset_done_agents(np.flatnonzero(done_mask))
                            if self._auto_reset_done
                            else {}
                        )
                    for agent_idx in np.flatnonzero(done_mask):
                        info = dict(done_infos[int(agent_idx)])
                        if self._auto_reset_done:
                            info["reset_info"] = reset_infos[int(agent_idx)]
                        else:
                            self._last_observation_infos[agent_idx] = dict(info)
                        infos[agent_idx] = info
            else:
                with self.gpu_render_profiler.time("latency_begin_s"):
                    self.latency.begin_decision(actions, active_mask=active)
                raw_frames_advanced = np.zeros(self.num_agents, dtype=np.int64)
                timeout_mask = np.zeros(self.num_agents, dtype=np.bool_)
                still_active = active.copy()
                latest_numeric_obs = self.core.observations()
                latest_scores = self._core_scores_numpy()
                for _ in range(self.clock.obs_stride_raw_frames):
                    with self.gpu_render_profiler.time("latency_raw_actions_s"):
                        raw_actions = self.latency.raw_actions(still_active)
                    with self.gpu_render_profiler.time("core_step_s"):
                        step_result = self.core.step_batch(raw_actions, active_mask=still_active)
                        step_rewards = step_result.rewards
                        step_terminated = step_result.terminated
                        step_truncated = step_result.truncated
                        latest_numeric_obs = step_result.observations
                        latest_scores = step_result.scores
                    advanced = still_active.copy()
                    raw_frames_advanced[advanced] += 1
                    self._raw_episode_steps[advanced] += 1
                    rewards[advanced] += step_rewards[advanced].astype(np.float32)
                    terminated |= step_terminated
                    truncated |= step_truncated
                    if self._batched_max_episode_steps > 0:
                        timeout = advanced & (self._raw_episode_steps >= self._batched_max_episode_steps)
                        truncated[timeout] = True
                        timeout_mask |= timeout
                    with self.gpu_render_profiler.time("latency_record_s"):
                        self.latency.record_raw_step(advanced)
                    still_active &= ~np.logical_or(terminated, truncated)
                    if not np.any(still_active):
                        break

                with self.gpu_render_profiler.time("latency_finish_s"):
                    latency_result = self.latency.finish_decision(active, raw_frames_advanced)
                    latency_raw_frames = latency_result.latency_raw_frames
                self._record_decision_history(
                    latency_result.command_admitted, issued_actions, latency_result.applied_actions, active
                )
                with self.gpu_render_profiler.time("metrics_update_s"):
                    done_infos = self.metrics.update(
                        issued_actions=issued_actions,
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        scores=latest_scores,
                        active_mask=active,
                        latency_raw_frames=latency_raw_frames,
                        command_admitted=latency_result.command_admitted,
                        timeout_mask=timeout_mask,
                    )

                done_mask = active & np.logical_or(terminated, truncated)
                if self.latency.add_latency_info:
                    for agent_idx in np.flatnonzero(active):
                        done = bool(done_mask[agent_idx])
                        latency_info = self.latency.decision_info_for_agent(int(agent_idx), latency_result)
                        info = {
                            "score": int(latest_scores[agent_idx]),
                            **latency_info,
                            "raw_reward": float(rewards[agent_idx]),
                        }
                        if bool(timeout_mask[agent_idx]):
                            info["TimeLimit.truncated"] = True
                        if int(agent_idx) in done_infos:
                            info.update(done_infos[int(agent_idx)])
                        info = self._with_numeric_obs(info, latest_numeric_obs[agent_idx])
                        if not (done and self._auto_reset_done):
                            self._last_observation_infos[agent_idx] = dict(info)
                        infos[agent_idx] = info
                    if self._auto_reset_done:
                        with self.gpu_render_profiler.time("auto_reset_s"):
                            reset_infos = self._auto_reset_done_agents(np.flatnonzero(done_mask))
                        for agent_idx, reset_info in reset_infos.items():
                            infos[agent_idx]["reset_info"] = reset_info
                else:
                    with self.gpu_render_profiler.time("auto_reset_s"):
                        reset_infos = (
                            self._auto_reset_done_agents(np.flatnonzero(done_mask))
                            if self._auto_reset_done
                            else {}
                        )
                    for agent_idx in np.flatnonzero(done_mask):
                        info = dict(done_infos[int(agent_idx)])
                        if self._auto_reset_done:
                            info["reset_info"] = reset_infos[int(agent_idx)]
                        else:
                            self._last_observation_infos[agent_idx] = dict(info)
                        infos[agent_idx] = info

        with self.gpu_render_profiler.time("step_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            self._append_env_raw_rgb_frames(raw_rgb)
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.step(
                    raw_rgb,
                    dones=np.logical_or(terminated, truncated),
                    active_mask=self._active_agents,
                )
        self.gpu_render_profiler.observe_step()
        self._attach_env_raw_rgb_infos(infos)
        self.sample_factory_step_cpu_tensors = {
            "rewards": torch.from_numpy(rewards),
            "terminated": torch.from_numpy(terminated),
            "truncated": torch.from_numpy(truncated),
            "dones": torch.from_numpy(np.logical_or(terminated, truncated)),
        }
        if getattr(self.cfg, "mask_unadmitted_actor", False):
            admitted = fused_result.command_admitted if use_cuda_latency else latency_result.command_admitted
            # Use the pre-reset active mask so terminal decisions keep their admission status.
            self.sample_factory_step_cpu_tensors["command_admitted"] = torch.from_numpy(admitted & active)
        return self._obs(obs), rewards_out, terminated_out, truncated_out, infos

    def _reset_env_raw_rgb_frames(self, raw_rgb: torch.Tensor, agent_indices: Sequence[int] | None = None) -> None:
        if not self.export_env_raw_rgb_frames:
            return
        frames = raw_rgb.detach().cpu().numpy()
        indices = range(self.num_agents) if agent_indices is None else [int(item) for item in agent_indices]
        for agent_idx in indices:
            self._env_raw_rgb_frames[agent_idx].clear()
            frame = np.array(frames[agent_idx], copy=True)
            for _ in range(self.observation_config.stack_frames):
                self._env_raw_rgb_frames[agent_idx].append(frame.copy())

    def _append_env_raw_rgb_frames(self, raw_rgb: torch.Tensor) -> None:
        if not self.export_env_raw_rgb_frames:
            return
        frames = raw_rgb.detach().cpu().numpy()
        for agent_idx in range(self.num_agents):
            self._env_raw_rgb_frames[agent_idx].append(np.array(frames[agent_idx], copy=True))

    def _attach_env_raw_rgb_infos(self, infos: list[dict[str, Any]]) -> None:
        if not self.export_env_raw_rgb_frames:
            return
        for agent_idx, info in enumerate(infos):
            updated = dict(info)
            updated[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = np.stack(tuple(self._env_raw_rgb_frames[agent_idx]), axis=0)
            infos[agent_idx] = updated

    def _auto_reset_done_agents(self, done_indices: Sequence[int]) -> dict[int, dict[str, Any]]:
        reset_indices = [int(agent_idx) for agent_idx in done_indices]
        if not reset_indices:
            return {}
        reset_seeds, reset_latency_infos = self.latency.reset(indices=reset_indices)
        self.metrics.reset(indices=reset_indices)
        reset_numeric_obs, reset_core_infos = self.core.reset_batch(
            seeds=reset_seeds,
            indices=reset_indices,
        )
        if self.decision_action_history is not None:
            self.decision_action_history.reset(reset_indices)
        reset_infos: dict[int, dict[str, Any]] = {}
        for offset, agent_idx in enumerate(reset_indices):
            self._raw_episode_steps[agent_idx] = 0
            self._raw_episode_steps_gpu[agent_idx] = 0
            reset_info = self._with_numeric_obs(
                {**reset_core_infos[agent_idx], **reset_latency_infos[offset]},
                reset_numeric_obs[agent_idx],
            )
            reset_infos[agent_idx] = reset_info
            self._last_observation_infos[agent_idx] = dict(reset_info)
        return reset_infos

    def _obs(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        expected_batch = self.num_agents
        if obs.ndim != 4 or int(obs.shape[0]) != expected_batch:
            raise RuntimeError(
                f"Expected batched observation tensor with shape ({expected_batch},C,H,W), got {tuple(obs.shape)}"
            )
        result = {"obs": obs}
        if self.decision_action_history is not None:
            result["transport"] = torch.as_tensor(self.decision_action_history.observation(), device=self.render_device)
        return result

    def _record_decision_history(self, admitted, issued, applied, active) -> None:
        if self.decision_action_history is None:
            return
        indices = np.flatnonzero(active)
        self.decision_action_history.append(indices, admitted[indices], issued[indices], applied[indices])

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

    def _normalize_actions(self, action: Any) -> list[Any]:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray):
            arr = action
            if arr.ndim == 0 and self.num_agents == 1:
                return [int(arr.item())]
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            if arr.ndim != 1 or int(arr.shape[0]) != self.num_agents:
                raise ValueError(f"Expected action array with shape ({self.num_agents},), got {arr.shape!r}")
            return [int(item) for item in arr.tolist()]
        if isinstance(action, (list, tuple)):
            if len(action) != self.num_agents:
                raise ValueError(f"Expected {self.num_agents} actions, got {len(action)}")
            return [self._scalar_action(item) for item in action]
        if self.num_agents == 1:
            return [self._scalar_action(action)]
        raise ValueError(f"Expected batched action for {self.num_agents} agents, got {type(action).__name__}")

    def _normalize_cuda_actions(self, action: Any) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            actions = action.to(device=self.render_device, dtype=torch.int64)
        else:
            actions = torch.as_tensor(action, dtype=torch.int64, device=self.render_device)
        if actions.ndim == 0:
            actions = actions.reshape(1)
        if actions.ndim == 2 and int(actions.shape[1]) == 1:
            actions = actions[:, 0]
        return actions.reshape(-1)

    @staticmethod
    def _scalar_action(action: Any) -> Any:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray):
            if action.size != 1:
                raise ValueError(f"Expected scalar action, got array with shape {action.shape!r}")
            return int(action.reshape(-1)[0])
        if isinstance(action, (list, tuple)) and len(action) == 1:
            return FlappyBatchedGpuRenderedEnv._scalar_action(action[0])
        return int(action)

    @staticmethod
    def _with_numeric_obs(info: dict[str, Any] | None, numeric_obs: Any) -> dict[str, Any]:
        out = dict(info or {})
        out["numeric_obs"] = numeric_obs
        return out

    def _core_scores_numpy(self) -> np.ndarray:
        score = self.core.score
        if isinstance(score, torch.Tensor):
            return score.detach().cpu().numpy().copy()
        return score.copy()
