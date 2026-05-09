from __future__ import annotations

from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict as GymDict

from modeling.flappy_ppo.observation import FlappyPpoObservationConfig
from modeling.flappy_ppo.preprocessing import FlappyPpoTorchObservationPipeline
from training.flappy_ppo.paths import ensure_flappy_gymnasium_on_path
from training.sample_factory_common.common import parse_int_list
from training.sample_factory_common.latency_wrapper import EpisodeMetricsWrapper, QueueLatencyWrapper, RewardTraceWrapper

from training.flappy_sf.gpu_rendering import _GpuRenderProfiler, resolve_gpu_render_device

ensure_flappy_gymnasium_on_path()

import flappy_bird_gymnasium  # noqa: F401,E402
from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState
from flappy_bird_gymnasium.gpu_render import FlappyRenderAssets, TorchFlappyRenderer


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

        self.render_device = resolve_gpu_render_device(gpu_render_device)
        self.observation_config = FlappyPpoObservationConfig(
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

        self.sub_envs = [self._make_sub_env(agent_idx) for agent_idx in range(self.num_agents)]
        self.action_space = self.sub_envs[0].action_space
        self.pipeline = FlappyPpoTorchObservationPipeline(
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

    def _make_sub_env(self, agent_idx: int) -> gym.Env:
        kwargs: dict[str, Any] = {
            "use_lidar": bool(getattr(self.cfg, "use_lidar", False)),
            "normalize_obs": bool(getattr(self.cfg, "normalize_obs", True)),
            "audio_on": bool(getattr(self.cfg, "audio_on", False)),
            "pipe_gap": int(getattr(self.cfg, "pipe_gap", 100)),
            "bird_color": getattr(self.cfg, "bird_color", "yellow"),
            "pipe_color": getattr(self.cfg, "pipe_color", "green"),
            "background": getattr(self.cfg, "background", "day"),
            "debug": bool(getattr(self.cfg, "debug", False)),
            "render_mode": None,
        }
        screen_size = getattr(self.cfg, "screen_size", None)
        if screen_size not in (None, ""):
            raise ValueError(
                "screen_size is not supported for Flappy Sample Factory; "
                "use env.obs_resize to resize observations without changing env geometry"
            )
        score_limit = getattr(self.cfg, "score_limit", None)
        if score_limit not in (None, "", 0, -1):
            kwargs["score_limit"] = int(score_limit)

        env = gym.make(getattr(self.cfg, "gym_id", "FlappyBird-v0"), **kwargs)
        max_episode_steps = int(getattr(self.cfg, "max_episode_steps", 0) or 0)
        if max_episode_steps > 0:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = RewardTraceWrapper(env)

        latency_values = parse_int_list(getattr(self.cfg, "queue_latency_values", "0"))
        sub_seed = self._sub_seed(agent_idx)
        env = QueueLatencyWrapper(
            env,
            queue_latency_values=latency_values or [0],
            latency_distribution_spec=getattr(self.cfg, "latency_distribution_spec", None) or None,
            latency_mode=getattr(self.cfg, "latency_mode", "episode_fixed"),
            latency_sampling_mode=getattr(self.cfg, "latency_sampling_mode", "uniform_all"),
            episodes_per_stage=int(getattr(self.cfg, "episodes_per_stage", 10)),
            randomize_within_stage=bool(getattr(self.cfg, "randomize_within_stage", False)),
            within_episode_change_interval=int(getattr(self.cfg, "within_episode_change_interval", 1)),
            within_episode_change_prob=float(getattr(self.cfg, "within_episode_change_prob", 1.0)),
            seed=sub_seed,
            env_reset_seed_base=sub_seed + 104729,
            latency_info_keys=("queue_latency", "frame_skip", "latency_frame_skip"),
            add_latency_info=bool(getattr(self.cfg, "add_latency_info", False)),
            max_pending_actions=getattr(self.cfg, "max_pending_actions", None),
        )
        env = EpisodeMetricsWrapper(
            env,
            frames_per_decision=1,
            metrics_path=str(getattr(self.cfg, "episode_metrics_path", "") or ""),
            worker_index=self._worker_index,
            vector_index=self._vector_index,
            env_id=self._sub_env_id(agent_idx),
            issued_action_labels={0: "noop", 1: "flap"},
        )
        return env

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        infos: list[dict[str, Any]] = []
        for agent_idx, env in enumerate(self.sub_envs):
            reset_kwargs = dict(kwargs)
            if seed is not None:
                reset_kwargs["seed"] = int(seed) + int(agent_idx)
            numeric_obs, info = env.reset(**reset_kwargs)
            infos.append(self._with_numeric_obs(info, numeric_obs))

        with self.gpu_render_profiler.time("reset_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.reset(raw_rgb, frame_skips=self._frame_skips(infos))
        self._has_reset = True
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
            for raw_agent_idx, raw_seed in reset_plan.items():
                agent_idx = int(raw_agent_idx)
                if agent_idx < 0 or agent_idx >= self.num_agents:
                    raise IndexError(f"agent index out of range: {agent_idx}")
                numeric_obs, info = self.sub_envs[agent_idx].reset(seed=int(raw_seed))
                info = self._with_numeric_obs(info, numeric_obs)
                infos[agent_idx] = info
                self._last_observation_infos[agent_idx] = dict(info)
                reset_mask[agent_idx] = True

        with self.gpu_render_profiler.time("reset_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.reset_indices(
                    raw_rgb,
                    reset_mask=reset_mask,
                    frame_skips=self._frame_skips(infos),
                )
        return self._obs(obs), infos

    def step(self, action):
        if not self._has_reset:
            self.reset()

        actions = self._normalize_actions(action)
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminated = np.zeros(self.num_agents, dtype=np.bool_)
        truncated = np.zeros(self.num_agents, dtype=np.bool_)
        infos: list[dict[str, Any]] = []
        observation_infos: list[dict[str, Any]] = []

        with self.gpu_render_profiler.time("env_step_s"):
            for agent_idx, (env, scalar_action) in enumerate(zip(self.sub_envs, actions)):
                if not bool(self._active_agents[agent_idx]):
                    observation_info = dict(self._last_observation_infos[agent_idx])
                    observation_info["inactive"] = True
                    infos.append(observation_info)
                    observation_infos.append(observation_info)
                    continue

                numeric_obs, reward, term, trunc, info = env.step(scalar_action)
                info = self._with_numeric_obs(info, numeric_obs)
                done = bool(term or trunc)
                rewards[agent_idx] = float(reward)
                terminated[agent_idx] = bool(term)
                truncated[agent_idx] = bool(trunc)
                if done and self._auto_reset_done:
                    reset_numeric_obs, reset_info = env.reset()
                    reset_info = self._with_numeric_obs(reset_info, reset_numeric_obs)
                    info["reset_info"] = reset_info
                    observation_infos.append(reset_info)
                    self._last_observation_infos[agent_idx] = dict(reset_info)
                else:
                    observation_infos.append(info)
                    self._last_observation_infos[agent_idx] = dict(info)
                infos.append(info)

        with self.gpu_render_profiler.time("step_obs_total_s"):
            raw_rgb = self._render_raw_rgb()
            with self.gpu_render_profiler.time("pipeline_s"):
                obs = self.pipeline.step(
                    raw_rgb,
                    dones=np.logical_or(terminated, truncated),
                    frame_skips=self._frame_skips(observation_infos),
                    active_mask=self._active_agents,
                )
        self.gpu_render_profiler.observe_step()
        return self._obs(obs), rewards, terminated, truncated, infos

    def _obs(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        expected_batch = self.num_agents
        if obs.ndim != 4 or int(obs.shape[0]) != expected_batch:
            raise RuntimeError(
                f"Expected batched observation tensor with shape ({expected_batch},C,H,W), got {tuple(obs.shape)}"
            )
        return {"obs": obs}

    def _render_raw_rgb(self) -> torch.Tensor:
        with self.gpu_render_profiler.time("get_render_state_s"):
            states = [env.unwrapped.get_render_state() for env in self.sub_envs]
        if self.renderer is None:
            assets = FlappyRenderAssets.for_state(states[0])
            self.renderer = TorchFlappyRenderer(assets, device=self.render_device)
        with self.gpu_render_profiler.time("state_batch_from_states_s"):
            batch = BatchedFlappyRenderState.from_states(states)
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

    @staticmethod
    def _frame_skips(infos: Sequence[dict[str, Any]]) -> np.ndarray:
        values = []
        for info in infos:
            values.append(int(info.get("frame_skip", info.get("latency_frame_skip", info.get("queue_latency", 0)))))
        return np.asarray(values, dtype=np.int64)

    def close(self):
        for env in self.sub_envs:
            env.close()
        super().close()
