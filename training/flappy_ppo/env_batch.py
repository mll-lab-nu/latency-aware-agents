from __future__ import annotations

from bisect import insort_right
from contextlib import nullcontext
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    derive_latency_seed,
    make_latency_distribution_sampler,
)
from training.flappy_ppo.paths import ensure_flappy_gymnasium_on_path


ensure_flappy_gymnasium_on_path()

import flappy_bird_gymnasium  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
from flappy_bird_gymnasium.envs.render_state import (  # noqa: E402
    BatchedFlappyRenderState,
)
from flappy_bird_gymnasium.gpu_render import (  # noqa: E402
    FlappyRenderAssets,
    TorchFlappyRenderer,
)


class QueueLatencyWrapper(gym.Wrapper):
    """True input-delay wrapper matching the SB3 queue-latency scripts."""

    def __init__(
        self,
        env: gym.Env,
        fixed_frame_skip: int,
        seed: int,
        latency_distribution_spec: LatencyDistributionSpec | None = None,
    ):
        super().__init__(env)
        self.fixed_frame_skip = int(fixed_frame_skip)
        self.latency_distribution_spec = latency_distribution_spec
        self.base_latency_seed = int(seed)
        self.latency_distribution_sampler = None
        self.current_frame_skip = 0
        self.current_applied_action = 0
        self.episode_idx = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions: List[tuple[int, int, int, int]] = []
        self._schedule_seq = 0

    def _reset_latency_distribution_sampler(self) -> None:
        if self.latency_distribution_spec is None:
            self.latency_distribution_sampler = None
            return
        self.latency_distribution_sampler = make_latency_distribution_sampler(
            self.latency_distribution_spec,
            seed=derive_latency_seed(self.base_latency_seed, episode_idx=int(self.episode_idx)),
        )

    def _next_frame_skip(self) -> int:
        if self.latency_distribution_spec is not None:
            if self.latency_distribution_sampler is None:
                self._reset_latency_distribution_sampler()
            return int(self.latency_distribution_sampler.sample_frame())
        return int(self.fixed_frame_skip)

    def _schedule_action(self, action: int) -> int:
        latency_for_action = int(self.current_frame_skip)
        apply_step = self.env_step_idx + latency_for_action
        self._schedule_seq += 1
        insort_right(
            self.pending_actions,
            (apply_step, self._schedule_seq, int(action), latency_for_action),
        )
        return apply_step

    def _advance_pending_actions(self) -> None:
        while self.pending_actions and self.pending_actions[0][0] <= self.env_step_idx:
            _, _, matured_action, _ = self.pending_actions.pop(0)
            self.current_applied_action = int(matured_action)

    def reset(self, **kwargs):
        self.current_applied_action = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions.clear()
        self._schedule_seq = 0
        self._reset_latency_distribution_sampler()
        self.current_frame_skip = 0 if self.latency_distribution_spec is not None else int(self.fixed_frame_skip)
        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        info["frame_skip"] = int(self.current_frame_skip)
        info["episode_idx"] = int(self.episode_idx)
        info["applied_action"] = int(self.current_applied_action)
        self.episode_idx += 1
        return obs, info

    def step(self, action):
        self.current_frame_skip = int(self._next_frame_skip())
        scheduled_delay = int(self.current_frame_skip)
        scheduled_apply_step = self._schedule_action(int(action))
        self._advance_pending_actions()
        obs, reward, terminated, truncated, info = self.env.step(
            int(self.current_applied_action)
        )
        info = dict(info or {})
        info["frame_skip"] = scheduled_delay
        info["decision_step_idx"] = int(self.decision_step_idx)
        info["env_step_idx"] = int(self.env_step_idx)
        info["applied_action"] = int(self.current_applied_action)
        info["queued_action"] = int(action)
        info["scheduled_apply_step"] = int(scheduled_apply_step)
        info["pending_queue_len"] = int(len(self.pending_actions))
        self.env_step_idx += 1
        self.decision_step_idx += 1
        return obs, float(reward), terminated, truncated, info


class SyncFlappyBatch:
    def __init__(
        self,
        n_envs: int,
        seed: int,
        fixed_frame_skip: int,
        device: str | torch.device,
        latency_distribution_spec: LatencyDistributionSpec | None = None,
    ) -> None:
        self.envs = []
        self.device = torch.device(device)
        self.episode_returns = np.zeros(n_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int32)
        first_state = None
        for idx in range(n_envs):
            env = gym.make(
                "FlappyBird-v0",
                use_lidar=False,
                normalize_obs=True,
                render_mode=None,
            )
            env = QueueLatencyWrapper(
                env,
                fixed_frame_skip=fixed_frame_skip,
                latency_distribution_spec=latency_distribution_spec,
                seed=seed + idx,
            )
            env.reset(seed=seed + idx)
            if first_state is None:
                first_state = env.unwrapped.get_render_state()
            self.envs.append(env)
        if first_state is None:
            raise ValueError("n_envs must be positive")
        assets = FlappyRenderAssets.for_state(first_state)
        self.renderer = TorchFlappyRenderer(assets, device=self.device)
        self._raw_rgb_buffer: torch.Tensor | None = None

    @property
    def n_envs(self) -> int:
        return len(self.envs)

    def reset(self) -> torch.Tensor:
        for idx, env in enumerate(self.envs):
            env.reset()
            self.episode_returns[idx] = 0.0
            self.episode_lengths[idx] = 0
        return self.render_raw_rgb()

    def step(self, actions: np.ndarray, profiler: Any | None = None):
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.bool_)
        infos: List[Dict] = []
        for idx, (env, action) in enumerate(zip(self.envs, actions)):
            _, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            self.episode_returns[idx] += float(reward)
            self.episode_lengths[idx] += 1
            if done:
                info["episodic_return"] = float(self.episode_returns[idx])
                info["episodic_length"] = int(self.episode_lengths[idx])
                env.reset()
                self.episode_returns[idx] = 0.0
                self.episode_lengths[idx] = 0
            rewards[idx] = float(reward)
            dones[idx] = done
            infos.append(info)
        return rewards, dones, infos, self.render_raw_rgb(profiler=profiler)

    def render_raw_rgb(self, profiler: Any | None = None) -> torch.Tensor:
        pack_timer = (
            profiler.time("state_batch_pack_s") if profiler is not None else nullcontext()
        )
        with pack_timer:
            batch = BatchedFlappyRenderState.from_states(self.render_states())
        render_timer = (
            profiler.time("raw_rgb_render_s") if profiler is not None else nullcontext()
        )
        with render_timer:
            return self.renderer.render_into(batch, self._raw_rgb_output_buffer(batch))

    def _raw_rgb_output_buffer(
        self,
        batch: BatchedFlappyRenderState,
    ) -> torch.Tensor:
        shape = (
            int(batch.batch_size),
            int(batch.screen_height),
            int(batch.screen_width),
            3,
        )
        expected_device = self.renderer.device
        if (
            self._raw_rgb_buffer is None
            or tuple(self._raw_rgb_buffer.shape) != shape
            or self._raw_rgb_buffer.device != expected_device
        ):
            self._raw_rgb_buffer = torch.empty(
                shape,
                dtype=torch.uint8,
                device=expected_device,
            )
        return self._raw_rgb_buffer

    def render_states(self):
        return [env.unwrapped.get_render_state() for env in self.envs]

    def render_frames(self):
        return list(self.render_raw_rgb().detach().cpu().numpy())

    def frame_skips(self) -> np.ndarray:
        return np.asarray(
            [env.current_frame_skip for env in self.envs], dtype=np.int64
        )

    def close(self):
        for env in self.envs:
            env.close()
