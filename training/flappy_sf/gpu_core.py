from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from gymnasium.utils import seeding

from flappy_bird_gymnasium.envs.batched_core import BatchedFlappyStepResult
from flappy_bird_gymnasium.envs.constants import (
    BACKGROUND_WIDTH,
    BASE_WIDTH,
    PIPE_HEIGHT,
    PIPE_VEL_X,
    PIPE_WIDTH,
    PLAYER_MAX_VEL_Y,
    PLAYER_ROT_THR,
)
from flappy_bird_gymnasium.envs.render_state import BatchedFlappyRenderState
from training.common.utils import batch_indices, seed_values
from training.flappy_sf.flappy_cuda_extension import load_flappy_cuda_extension


@dataclass
class TensorFlappyStepResult:
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    scores: torch.Tensor


@dataclass
class NumpyFlappyStepResult:
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    scores: np.ndarray


class CudaFlappyBirdCore:
    def __init__(
        self,
        batch_size: int,
        *,
        screen_size: tuple[int, int] = (288, 512),
        normalize_obs: bool = True,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        background: str | None = "day",
        score_limit: int | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.screen_width = int(screen_size[0])
        self.screen_height = int(screen_size[1])
        self.normalize_obs = bool(normalize_obs)
        self.pipe_gap = int(pipe_gap)
        self.bird_color = bird_color
        self.pipe_color = pipe_color
        self.background = background
        score_limit_value = -1 if score_limit is None else int(score_limit)
        self.score_limit = torch.full((self.batch_size,), score_limit_value, dtype=torch.int64, device=self.device)

        self.player_x = torch.full((self.batch_size,), int(self.screen_width * 0.2), dtype=torch.float64, device=self.device)
        self.player_y = torch.zeros(self.batch_size, dtype=torch.float64, device=self.device)
        self.player_vel_y = torch.zeros(self.batch_size, dtype=torch.float64, device=self.device)
        self.player_rot = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.player_idx = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self._player_idx_cycle_pos = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.loop_iter = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.score = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.ground_x = torch.zeros(self.batch_size, dtype=torch.float64, device=self.device)
        self.ground_y = torch.full((self.batch_size,), self.screen_height * 0.79, dtype=torch.float64, device=self.device)
        self.upper_pipe_xy = torch.zeros((self.batch_size, 3, 2), dtype=torch.float64, device=self.device)
        self.lower_pipe_xy = torch.zeros((self.batch_size, 3, 2), dtype=torch.float64, device=self.device)
        self.np_randoms = [seeding.np_random(None)[0] for _ in range(self.batch_size)]
        self._base_shift = BASE_WIDTH - BACKGROUND_WIDTH
        if self.device.type != "cuda":
            raise RuntimeError("CudaFlappyBirdCore requires a CUDA device")
        self.ext = load_flappy_cuda_extension()
        self._rewards = torch.empty(self.batch_size, dtype=torch.float64, device=self.device)
        self._terminated = torch.empty(self.batch_size, dtype=torch.bool, device=self.device)
        self._truncated = torch.empty(self.batch_size, dtype=torch.bool, device=self.device)
        self._offscreen = torch.empty((self.batch_size, 3), dtype=torch.bool, device=self.device)
        self._packed_step = torch.empty((self.batch_size, 7), dtype=torch.float64, device=self.device)
        self._packed_decision = torch.empty((self.batch_size, 9), dtype=torch.float64, device=self.device)
        self._decision_rewards = torch.empty(self.batch_size, dtype=torch.float32, device=self.device)
        self._decision_terminated = torch.empty(self.batch_size, dtype=torch.bool, device=self.device)
        self._decision_truncated = torch.empty(self.batch_size, dtype=torch.bool, device=self.device)
        self._pipe_reset_slot_capacity = 1
        self._pipe_reset_upper_xy = torch.empty((self.batch_size, 3, 1, 2), dtype=torch.float64, device=self.device)
        self._pipe_reset_lower_xy = torch.empty((self.batch_size, 3, 1, 2), dtype=torch.float64, device=self.device)
        self._pipe_reset_upper_xy_cpu = np.empty((self.batch_size, 3, 1, 2), dtype=np.float64)
        self._pipe_reset_lower_xy_cpu = np.empty((self.batch_size, 3, 1, 2), dtype=np.float64)

    def render_state_batch(self) -> BatchedFlappyRenderState:
        return BatchedFlappyRenderState(
            player_xy=torch.stack([self.player_x, self.player_y], dim=1).to(torch.float32),
            player_indices=self.player_idx.to(torch.int64),
            visible_rotations=self.visible_rotations_tensor(),
            upper_pipe_xy=self.upper_pipe_xy.to(torch.float32),
            lower_pipe_xy=self.lower_pipe_xy.to(torch.float32),
            ground_xy=torch.stack([self.ground_x, self.ground_y], dim=1).to(torch.float32),
            screen_width=int(self.screen_width),
            screen_height=int(self.screen_height),
            bird_color=self.bird_color,
            pipe_color=self.pipe_color,
            background=self.background,
        )

    def visible_rotations_tensor(self) -> torch.Tensor:
        visible_rotations = torch.full((self.batch_size,), PLAYER_ROT_THR, dtype=torch.int16, device=self.device)
        mask = self.player_rot <= PLAYER_ROT_THR
        visible_rotations[mask] = self.player_rot[mask].to(torch.int16)
        return visible_rotations

    def _random_pipe(self, agent_idx: int) -> tuple[tuple[float, float], tuple[float, float]]:
        gap_ys = [20, 30, 40, 50, 60, 70, 80, 90]
        gap_y = gap_ys[int(self.np_randoms[agent_idx].integers(0, len(gap_ys)))]
        gap_y += int(float(self.ground_y[agent_idx].detach().cpu().item()) * 0.2)
        pipe_x = self.screen_width + PIPE_WIDTH + (self.screen_width * 0.2)
        return (
            (float(pipe_x), float(gap_y - PIPE_HEIGHT)),
            (float(pipe_x), float(gap_y + self.pipe_gap)),
        )

    def _indices(self, indices: Sequence[int] | np.ndarray | torch.Tensor | None) -> list[int]:
        return batch_indices(indices, batch_size=self.batch_size)

    @staticmethod
    def _seed_values(seeds: Sequence[int | None] | int | None, count: int) -> list[int | None]:
        return seed_values(seeds, count)

    def reset_batch(
        self,
        seeds: Sequence[int | None] | int | None = None,
        indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        target_indices = self._indices(indices)
        seed_values = self._seed_values(seeds, len(target_indices))
        upper_reset = np.empty((len(target_indices), 3, 2), dtype=np.float64)
        lower_reset = np.empty((len(target_indices), 3, 2), dtype=np.float64)
        for offset, agent_idx in enumerate(target_indices):
            seed = seed_values[offset]
            if seed is not None:
                self.np_randoms[agent_idx] = seeding.np_random(int(seed))[0]
            pipe1 = self._random_pipe(agent_idx)
            pipe2 = self._random_pipe(agent_idx)
            pipe3 = self._random_pipe(agent_idx)
            upper_reset[offset] = np.asarray(
                [
                    [self.screen_width, pipe1[0][1]],
                    [self.screen_width + (self.screen_width / 2), pipe2[0][1]],
                    [self.screen_width + self.screen_width, pipe3[0][1]],
                ],
                dtype=np.float64,
            )
            lower_reset[offset] = np.asarray(
                [
                    [self.screen_width, pipe1[1][1]],
                    [self.screen_width + (self.screen_width / 2), pipe2[1][1]],
                    [self.screen_width + self.screen_width, pipe3[1][1]],
                ],
                dtype=np.float64,
            )
        self.ext.flappy_reset(
            torch.as_tensor(target_indices, dtype=torch.int64, device=self.device).contiguous(),
            torch.as_tensor(upper_reset, dtype=torch.float64, device=self.device).contiguous(),
            torch.as_tensor(lower_reset, dtype=torch.float64, device=self.device).contiguous(),
            self.player_x,
            self.player_y,
            self.player_vel_y,
            self.player_rot,
            self.player_idx,
            self._player_idx_cycle_pos,
            self.loop_iter,
            self.score,
            self.ground_x,
            self.ground_y,
            self.upper_pipe_xy,
            self.lower_pipe_xy,
            int(len(target_indices)),
            int(self.screen_width),
            int(self.screen_height),
        )
        observations = self.observations()
        scores = self.score.detach().cpu().numpy()
        infos = [{"score": int(scores[agent_idx])} for agent_idx in range(self.batch_size)]
        return observations, infos

    def _ext_flappy_step(
        self,
        actions: Sequence[int] | np.ndarray | torch.Tensor,
        active_mask: Sequence[bool] | np.ndarray | torch.Tensor | None,
    ) -> None:
        action_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).reshape(-1).contiguous()
        active_t = (
            torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            if active_mask is None
            else torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1).contiguous()
        )
        self.ext.flappy_step(
            self.player_x,
            self.player_y,
            self.player_vel_y,
            self.player_rot,
            self.player_idx,
            self._player_idx_cycle_pos,
            self.loop_iter,
            self.score,
            self.score_limit,
            self.ground_x,
            self.ground_y,
            self.upper_pipe_xy,
            self.lower_pipe_xy,
            action_t,
            active_t,
            self._rewards,
            self._terminated,
            self._truncated,
            self._offscreen,
            self._packed_step,
            int(self.batch_size),
            int(self.screen_width),
            int(self.screen_height),
            int(self._base_shift),
        )

    def step_batch_tensors(
        self,
        actions: Sequence[int] | np.ndarray | torch.Tensor,
        active_mask: Sequence[bool] | np.ndarray | torch.Tensor | None = None,
    ) -> TensorFlappyStepResult:
        self._ext_flappy_step(actions, active_mask)
        self._reset_offscreen_pipes()
        return TensorFlappyStepResult(
            rewards=self._rewards,
            terminated=self._terminated,
            truncated=self._truncated,
            scores=self.score,
        )

    def step_batch_numpy(
        self,
        actions: Sequence[int] | np.ndarray | torch.Tensor,
        active_mask: Sequence[bool] | np.ndarray | torch.Tensor | None = None,
    ) -> NumpyFlappyStepResult:
        self._ext_flappy_step(actions, active_mask)
        packed = self._packed_step.detach().cpu().numpy()
        self._reset_offscreen_pipes_from_numpy(packed[:, 4:7])
        return NumpyFlappyStepResult(
            rewards=packed[:, 0],
            terminated=packed[:, 1].astype(np.bool_),
            truncated=packed[:, 2].astype(np.bool_),
            scores=packed[:, 3].astype(np.int64),
        )

    def step_batch(
        self,
        actions: Sequence[int] | np.ndarray | torch.Tensor,
        active_mask: Sequence[bool] | np.ndarray | torch.Tensor | None = None,
    ) -> BatchedFlappyStepResult:
        result = self.step_batch_tensors(actions, active_mask=active_mask)
        return BatchedFlappyStepResult(
            observations=self.observations(),
            rewards=result.rewards.detach().cpu().numpy().copy(),
            terminated=result.terminated.detach().cpu().numpy().copy(),
            truncated=result.truncated.detach().cpu().numpy().copy(),
            scores=result.scores.detach().cpu().numpy().copy(),
        )

    def prepare_pipe_resets_for_decision(
        self,
        active_mask: torch.Tensor,
        raw_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        raw_steps = int(raw_steps)
        if self._pipe_reset_slot_capacity < raw_steps:
            self._pipe_reset_slot_capacity = raw_steps
            self._pipe_reset_upper_xy = torch.empty(
                (self.batch_size, 3, self._pipe_reset_slot_capacity, 2),
                dtype=torch.float64,
                device=self.device,
            )
            self._pipe_reset_lower_xy = torch.empty_like(self._pipe_reset_upper_xy)
            self._pipe_reset_upper_xy_cpu = np.empty(
                (self.batch_size, 3, self._pipe_reset_slot_capacity, 2),
                dtype=np.float64,
            )
            self._pipe_reset_lower_xy_cpu = np.empty_like(self._pipe_reset_upper_xy_cpu)

        active = active_mask.reshape(-1).detach().cpu().numpy().astype(np.bool_, copy=False)
        x_positions = self.upper_pipe_xy[:, :, 0].detach().cpu().numpy().copy()
        reset_counts = np.zeros((self.batch_size, 3), dtype=np.int64)
        for _ in range(raw_steps):
            x_positions[active, :] += float(PIPE_VEL_X)
            for pipe_idx in range(3):
                for agent_idx in np.flatnonzero(active & (x_positions[:, pipe_idx] < -float(PIPE_WIDTH))):
                    slot_idx = int(reset_counts[int(agent_idx), pipe_idx])
                    upper, lower = self._random_pipe(int(agent_idx))
                    self._pipe_reset_upper_xy_cpu[int(agent_idx), pipe_idx, slot_idx] = upper
                    self._pipe_reset_lower_xy_cpu[int(agent_idx), pipe_idx, slot_idx] = lower
                    reset_counts[int(agent_idx), pipe_idx] += 1
                    x_positions[int(agent_idx), pipe_idx] = upper[0]
        if np.any(reset_counts):
            self._pipe_reset_upper_xy.copy_(torch.as_tensor(self._pipe_reset_upper_xy_cpu, device=self.device))
            self._pipe_reset_lower_xy.copy_(torch.as_tensor(self._pipe_reset_lower_xy_cpu, device=self.device))
        return self._pipe_reset_upper_xy, self._pipe_reset_lower_xy, int(self._pipe_reset_slot_capacity)

    def observations(self) -> np.ndarray:
        pipes = np.empty((self.batch_size, 3, 3), dtype=np.float64)
        lower_pipe_xy = self.lower_pipe_xy.detach().cpu().numpy()
        upper_pipe_xy = self.upper_pipe_xy.detach().cpu().numpy()
        for pipe_idx in range(3):
            low_x = lower_pipe_xy[:, pipe_idx, 0]
            top_y = upper_pipe_xy[:, pipe_idx, 1] + PIPE_HEIGHT
            low_y = lower_pipe_xy[:, pipe_idx, 1]
            pipes[:, pipe_idx, 0] = low_x
            pipes[:, pipe_idx, 1] = top_y
            pipes[:, pipe_idx, 2] = low_y
            behind = low_x > self.screen_width
            pipes[behind, pipe_idx, 0] = self.screen_width
            pipes[behind, pipe_idx, 1] = 0
            pipes[behind, pipe_idx, 2] = self.screen_height

        order = np.argsort(pipes[:, :, 0], axis=1)
        sorted_pipes = np.take_along_axis(pipes, order[:, :, None], axis=1)
        pos_y = self.player_y.detach().cpu().numpy().copy()
        vel_y = self.player_vel_y.detach().cpu().numpy().copy()
        rot = self.player_rot.detach().cpu().numpy().astype(np.float64)
        if self.normalize_obs:
            sorted_pipes[:, :, 0] /= self.screen_width
            sorted_pipes[:, :, 1] /= self.screen_height
            sorted_pipes[:, :, 2] /= self.screen_height
            pos_y /= self.screen_height
            vel_y /= PLAYER_MAX_VEL_Y
            rot /= 90

        return np.concatenate(
            [
                sorted_pipes.reshape(self.batch_size, 9),
                pos_y[:, None],
                vel_y[:, None],
                rot[:, None],
            ],
            axis=1,
        )

    def _reset_offscreen_pipes(self) -> None:
        offscreen = self._offscreen.detach().cpu().numpy()
        self._reset_offscreen_pipes_from_numpy(offscreen)

    def _reset_offscreen_pipes_from_numpy(self, offscreen: np.ndarray) -> None:
        if not bool(np.any(offscreen)):
            return
        for agent_idx, pipe_idx in np.argwhere(offscreen):
            upper, lower = self._random_pipe(int(agent_idx))
            self.upper_pipe_xy[int(agent_idx), int(pipe_idx)] = torch.tensor(
                upper,
                dtype=torch.float64,
                device=self.device,
            )
            self.lower_pipe_xy[int(agent_idx), int(pipe_idx)] = torch.tensor(
                lower,
                dtype=torch.float64,
                device=self.device,
            )
