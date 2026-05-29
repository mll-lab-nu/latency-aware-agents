from __future__ import annotations

from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict as GymDict, Discrete

from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyBirdCore

from training.flappy_sf.batched_metrics import BatchedFlappyEpisodeMetrics
from training.flappy_sf.batched_queue_latency import BatchedQueueLatencyScheduler
from training.flappy_sf.gpu_rendering import (
    FlappyObservationConfig,
    FlappyTorchObservationPipeline,
    FlappyTorchRenderPipeline,
    GpuRenderProfiler,
    resolve_gpu_render_device,
)
from training.sample_factory_common.common import env_instance_seed, parse_int_list


class FlappyBatchedGpuRenderedEnv(gym.Env):
    """Sample Factory multi-agent Flappy env using batched core and Torch rendering."""

    is_multiagent = True

    def __init__(
        self,
        *,
        cfg: Any,
        env_config: Any | None,
        image_size: tuple[int, int],
        frame_stack: int,
        gpu_render_device: str | torch.device,
        gpu_render_batch_size: int,
        gpu_render_profile: bool,
        gpu_render_profile_interval: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.env_config = env_config
        self.num_agents = int(gpu_render_batch_size)
        height, width = int(image_size[1]), int(image_size[0])
        if height != width:
            raise ValueError("use_gpu_render=true currently requires square policy_screen_size")

        self.render_device = resolve_gpu_render_device(gpu_render_device)
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
        self.action_space = Discrete(2)
        self.uses_batched_core = True
        self.sub_envs: list[gym.Env] = []

        score_limit = getattr(self.cfg, "score_limit", None)
        resolved_score_limit = None if score_limit in (None, "", 0, -1) else int(score_limit)
        self.core = BatchedFlappyBirdCore(
            batch_size=self.num_agents,
            normalize_obs=bool(getattr(self.cfg, "normalize_obs", True)),
            pipe_gap=int(getattr(self.cfg, "pipe_gap", 100)),
            bird_color=getattr(self.cfg, "bird_color", "yellow"),
            pipe_color=getattr(self.cfg, "pipe_color", "green"),
            background=getattr(self.cfg, "background", "day"),
            score_limit=resolved_score_limit,
        )

        sub_seeds = [self._sub_seed(agent_idx) for agent_idx in range(self.num_agents)]
        latency_values = parse_int_list(getattr(self.cfg, "queue_latency_values", "0"))
        self.latency = BatchedQueueLatencyScheduler(
            batch_size=self.num_agents,
            queue_latency_values=latency_values or [0],
            latency_mode=getattr(self.cfg, "latency_mode", "episode_fixed"),
            latency_sampling_mode=getattr(self.cfg, "latency_sampling_mode", "uniform_all"),
            latency_distribution=getattr(self.cfg, "latency_distribution", "discrete"),
            latency_normal_mean=float(getattr(self.cfg, "latency_normal_mean", 0.0) or 0.0),
            latency_normal_std=float(getattr(self.cfg, "latency_normal_std", 0.0) or 0.0),
            latency_normal_min=int(getattr(self.cfg, "latency_normal_min", 0) or 0),
            latency_normal_max=int(getattr(self.cfg, "latency_normal_max", 0) or 0),
            latency_uniform_min=int(getattr(self.cfg, "latency_uniform_min", 0) or 0),
            latency_uniform_max=int(getattr(self.cfg, "latency_uniform_max", 0) or 0),
            episodes_per_stage=int(getattr(self.cfg, "episodes_per_stage", 10)),
            randomize_within_stage=bool(getattr(self.cfg, "randomize_within_stage", False)),
            within_episode_change_interval=int(getattr(self.cfg, "within_episode_change_interval", 1)),
            within_episode_change_prob=float(getattr(self.cfg, "within_episode_change_prob", 1.0)),
            seeds=sub_seeds,
            env_reset_seed_bases=[seed + 104729 for seed in sub_seeds],
            latency_info_keys=("queue_latency", "frame_skip", "latency_frame_skip"),
            add_latency_info=bool(getattr(self.cfg, "add_latency_info", False)),
        )
        self.metrics = BatchedFlappyEpisodeMetrics(
            batch_size=self.num_agents,
            metrics_path=str(getattr(self.cfg, "episode_metrics_path", "") or ""),
            worker_index=self._worker_index,
            vector_index=self._vector_index,
            env_ids=[self._sub_env_id(agent_idx) for agent_idx in range(self.num_agents)],
        )
        self.pipeline = FlappyTorchObservationPipeline(
            self.observation_config,
            n_envs=self.num_agents,
            device=self.render_device,
        )
        self.renderer = FlappyTorchRenderPipeline(device=self.render_device)
        self.gpu_render_profiler = GpuRenderProfiler(
            enabled=bool(gpu_render_profile),
            interval=int(gpu_render_profile_interval),
            device=self.render_device,
            worker_index=self._worker_index,
            vector_index=self._vector_index,
            env_id=self._env_id,
        )
        self._has_reset = False
        self._active_agents = np.ones(self.num_agents, dtype=np.bool_)
        self._auto_reset_done = True
        self._last_observation_infos: list[dict[str, Any]] = [{} for _ in range(self.num_agents)]
        self._decision_steps = np.zeros(self.num_agents, dtype=np.int64)
        self._max_episode_steps = int(getattr(self.cfg, "max_episode_steps", 0) or 0)

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
        return env_instance_seed(
            base_seed=int(getattr(self.cfg, "seed", 0) or 0),
            worker_index=self._worker_index,
            vector_index=self._vector_index,
            env_id=self._sub_env_id(agent_idx),
        )

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
        self._decision_steps[:] = 0
        numeric_obs, core_infos = self.core.reset_batch(seeds=reset_seeds)
        infos = [
            self._with_numeric_obs({**core_infos[agent_idx], **latency_infos[agent_idx]}, numeric_obs[agent_idx])
            for agent_idx in range(self.num_agents)
        ]
        with self.gpu_render_profiler.time("reset_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.reset(raw_rgb)
        self._has_reset = True
        self._last_observation_infos = [dict(info) for info in infos]
        return self._obs(obs), infos

    def set_auto_reset_done(self, enabled: bool) -> None:
        self._auto_reset_done = bool(enabled)

    def set_active_agents(self, active_mask: Sequence[bool]) -> None:
        self._active_agents = np.asarray(list(active_mask), dtype=np.bool_).reshape(-1)

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
        reset_indices = [int(raw_agent_idx) for raw_agent_idx in reset_plan.keys()]
        reset_seeds_input = [int(raw_seed) for raw_seed in reset_plan.values()]
        reset_seeds, latency_infos = self.latency.reset(indices=reset_indices, seeds=reset_seeds_input)
        self.metrics.reset(indices=reset_indices)
        numeric_obs, core_infos = self.core.reset_batch(seeds=reset_seeds, indices=reset_indices)
        reset_mask = np.zeros(self.num_agents, dtype=np.bool_)
        for offset, agent_idx in enumerate(reset_indices):
            info = self._with_numeric_obs(
                {**core_infos[agent_idx], **latency_infos[offset]},
                numeric_obs[agent_idx],
            )
            infos[agent_idx] = info
            self._last_observation_infos[agent_idx] = dict(info)
            self._decision_steps[agent_idx] = 0
            reset_mask[agent_idx] = True

        with self.gpu_render_profiler.time("reset_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.reset_indices(raw_rgb, reset_mask=reset_mask)
        return self._obs(obs), infos

    def step(self, action):
        if not self._has_reset:
            self.reset()

        actions = np.asarray(self._normalize_actions(action), dtype=np.int64)
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminated = np.zeros(self.num_agents, dtype=np.bool_)
        truncated = np.zeros(self.num_agents, dtype=np.bool_)
        infos: list[dict[str, Any]] = [{} for _agent_idx in range(self.num_agents)]
        active = self._active_agents.copy()

        with self.gpu_render_profiler.time("env_step_s"):
            for agent_idx in np.flatnonzero(~active):
                inactive_info = dict(self._last_observation_infos[agent_idx])
                inactive_info["inactive"] = True
                infos[agent_idx] = inactive_info

            self.latency.begin_decision(actions, active_mask=active)
            raw_actions = self.latency.raw_actions(active)
            step_result = self.core.step_batch(raw_actions, active_mask=active)
            rewards[active] = step_result.rewards[active].astype(np.float32)
            terminated |= step_result.terminated
            truncated |= step_result.truncated
            self._decision_steps[active] += 1
            if self._max_episode_steps > 0:
                truncated[active & (self._decision_steps >= self._max_episode_steps)] = True
            self.latency.record_step(active)
            latency_result = self.latency.finish_decision(active)
            queue_latencies = latency_result.scheduled_latencies.copy()
            done_infos = self.metrics.update(
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                active_mask=active,
                scores=step_result.scores,
                queue_latencies=queue_latencies,
            )

            done_mask = active & np.logical_or(terminated, truncated)
            for agent_idx in np.flatnonzero(active):
                info = {
                    "score": int(step_result.scores[agent_idx]),
                    **self.latency.decision_info_for_agent(int(agent_idx), latency_result),
                    "raw_reward": float(rewards[agent_idx]),
                    "transformed_reward": float(rewards[agent_idx]),
                }
                if int(agent_idx) in done_infos:
                    info.update(done_infos[int(agent_idx)])
                info = self._with_numeric_obs(info, step_result.observations[agent_idx])
                if bool(done_mask[agent_idx]) and self._auto_reset_done:
                    reset_seeds, reset_latency_infos = self.latency.reset(indices=[int(agent_idx)])
                    self.metrics.reset(indices=[int(agent_idx)])
                    reset_numeric_obs, reset_core_infos = self.core.reset_batch(
                        seeds=reset_seeds,
                        indices=[int(agent_idx)],
                    )
                    self._decision_steps[agent_idx] = 0
                    reset_info = self._with_numeric_obs(
                        {**reset_core_infos[agent_idx], **reset_latency_infos[0]},
                        reset_numeric_obs[agent_idx],
                    )
                    info["reset_info"] = reset_info
                    self._last_observation_infos[agent_idx] = dict(reset_info)
                else:
                    self._last_observation_infos[agent_idx] = dict(info)
                infos[agent_idx] = info

        with self.gpu_render_profiler.time("step_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.step(raw_rgb, dones=np.logical_or(terminated, truncated), active_mask=active)
        self.gpu_render_profiler.observe_step()
        return self._obs(obs), rewards, terminated, truncated, infos

    def render(self, agent_idx: int = 0):
        return self._render_raw_rgb().detach().cpu().numpy()[int(agent_idx)]

    def _obs(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"obs": obs}

    def _render_raw_rgb(self) -> torch.Tensor:
        with self.gpu_render_profiler.time("get_render_state_s"):
            batch = self.core.render_state_batch()
        with self.gpu_render_profiler.time("render_into_s"):
            return self.renderer.render(batch)

    def _normalize_actions(self, action: Any) -> list[Any]:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray):
            arr = action
            if arr.ndim == 0 and self.num_agents == 1:
                return [int(arr.item())]
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            return [int(item) for item in arr.reshape(-1).tolist()]
        if isinstance(action, (list, tuple)):
            return [self._scalar_action(item) for item in action]
        return [self._scalar_action(action)]

    @staticmethod
    def _scalar_action(action: Any) -> Any:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray):
            return int(action.reshape(-1)[0])
        if isinstance(action, (list, tuple)) and len(action) == 1:
            return FlappyBatchedGpuRenderedEnv._scalar_action(action[0])
        return int(action)

    @staticmethod
    def _with_numeric_obs(info: dict[str, Any] | None, numeric_obs: Any) -> dict[str, Any]:
        out = dict(info or {})
        out["numeric_obs"] = numeric_obs
        return out

    def close(self):
        super().close()
