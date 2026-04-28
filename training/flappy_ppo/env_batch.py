from __future__ import annotations

from bisect import insort_right
from contextlib import nullcontext
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

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

    def __init__(self, env: gym.Env, frame_skip_values: Sequence[int], seed: int):
        super().__init__(env)
        if not frame_skip_values:
            raise ValueError("frame_skip_values cannot be empty")
        self.frame_skip_values = [max(0, int(x)) for x in frame_skip_values]
        self.rng = np.random.RandomState(seed)
        self.current_frame_skip = 0
        self.current_applied_action = 0
        self.episode_idx = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions: List[tuple[int, int, int, int]] = []
        self._schedule_seq = 0

    def _sample_frame_skip(self) -> int:
        return int(self.rng.choice(self.frame_skip_values))

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
        self.current_frame_skip = self._sample_frame_skip()
        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        info["frame_skip"] = int(self.current_frame_skip)
        info["episode_idx"] = int(self.episode_idx)
        info["applied_action"] = int(self.current_applied_action)
        self.episode_idx += 1
        return obs, info

    def step(self, action):
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
        frame_skip_values: Sequence[int],
        device: str | torch.device,
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
            env = QueueLatencyWrapper(env, frame_skip_values, seed=seed + idx)
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
