from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Sequence

import numpy as np
import torch

from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
from training.demon_attack_sf.cule_atari_extension import load_cule_atari_extension
from training.demon_attack_sf.demon_attack_cuda_extension import load_demon_attack_cuda_extension


_DECISION_REWARD = 0
_DECISION_TERMINATED = 1
_DECISION_TRUNCATED = 2
_DECISION_SCORE = 3
_DECISION_RAW_FRAMES_ADVANCED = 4
_DECISION_LIVES = 5
_DECISION_ENV_STEP_IDX = 6
_DECISION_DECISION_STEP_IDX = 7
_DECISION_LATENCY_RAW_FRAMES = 8
_DECISION_APPLIED_ACTION = 9
_DECISION_PENDING_COUNT = 10
_DECISION_READY_RAW_FRAME = 11
_DECISION_QUEUED_ACTION = 12
_DECISION_TIMEOUT = 13
_DECISION_COLUMN_COUNT = 14


def load_demon_attack_rom_bytes() -> bytes:
    return resources.files("ale_py.roms").joinpath("demon_attack.bin").read_bytes()


def demon_attack_rom_path() -> str:
    return str(resources.files("ale_py.roms").joinpath("demon_attack.bin"))


@dataclass
class CudaDemonAttackStepResult:
    observations: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    scores: torch.Tensor
    lives: torch.Tensor


@dataclass
class CudaDemonAttackDecisionResult:
    observations: torch.Tensor
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    scores: np.ndarray
    lives: np.ndarray
    raw_frames_advanced: np.ndarray
    latency_raw_frames: np.ndarray
    queued_actions: np.ndarray
    applied_actions: np.ndarray
    ready_raw_frames: np.ndarray
    pending_counts: np.ndarray
    env_step_indices: np.ndarray
    decision_step_indices: np.ndarray
    timeout_mask: np.ndarray
    rewards_gpu: torch.Tensor
    terminated_gpu: torch.Tensor
    truncated_gpu: torch.Tensor


class CudaDemonAttackCore:
    def __init__(
        self,
        batch_size: int,
        *,
        device: str | torch.device = "cuda",
        noop_max: int = 30,
        terminal_on_life_loss: bool = False,
    ) -> None:
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.noop_max = int(noop_max)
        self.terminal_on_life_loss = bool(terminal_on_life_loss)
        self.ext = load_cule_atari_extension()
        self.kernels = load_demon_attack_cuda_extension()
        self.rom = self.ext.AtariRom(demon_attack_rom_path())
        self.env = self.ext.AtariEnv(self.rom, self.batch_size, self.noop_max)
        self.screen_height = 84
        self.screen_width = 84
        self.num_channels = 1
        self.minimal_action_values = [int(action) for action in self.rom.minimal_actions()]
        self.action_set = torch.tensor(self.minimal_action_values, dtype=torch.uint8, device=self.device)

        self.states = torch.zeros((self.batch_size, self.env.state_size()), dtype=torch.uint8, device=self.device)
        self.frame_states = torch.zeros(
            (self.batch_size, self.env.frame_state_size()),
            dtype=torch.uint8,
            device=self.device,
        )
        self.ram = torch.randint(0, 255, (self.batch_size, self.rom.ram_size()), dtype=torch.uint8, device=self.device)
        self.tia = torch.zeros((self.batch_size, self.env.tia_update_size()), dtype=torch.int32, device=self.device)
        self.frame_buffer = torch.zeros(
            (self.batch_size, 300 * self.rom.screen_width()),
            dtype=torch.uint8,
            device=self.device,
        )
        self.cart_offsets = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self.rand_states = torch.randint(
            0,
            np.iinfo(np.int32).max,
            (self.batch_size,),
            dtype=torch.int32,
            device=self.device,
        )
        self.cached_states = torch.zeros((self.noop_max, self.env.state_size()), dtype=torch.uint8, device=self.device)
        self.cached_ram = torch.randint(
            0,
            255,
            (self.noop_max, self.rom.ram_size()),
            dtype=torch.uint8,
            device=self.device,
        )
        self.cached_frame_states = torch.zeros(
            (self.noop_max, self.env.frame_state_size()),
            dtype=torch.uint8,
            device=self.device,
        )
        self.cached_tia = torch.zeros(
            (self.noop_max, self.env.tia_update_size()),
            dtype=torch.int32,
            device=self.device,
        )
        self.cache_index = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)

        self.env.set_cuda(True, self.device.index if self.device.index is not None else torch.cuda.current_device())
        self.env.initialize(
            self.states.data_ptr(),
            self.frame_states.data_ptr(),
            self.ram.data_ptr(),
            self.tia.data_ptr(),
            self.frame_buffer.data_ptr(),
            self.cart_offsets.data_ptr(),
            self.action_set.data_ptr(),
            self.rand_states.data_ptr(),
            self.cached_states.data_ptr(),
            self.cached_ram.data_ptr(),
            self.cached_frame_states.data_ptr(),
            self.cached_tia.data_ptr(),
            self.cache_index.data_ptr(),
        )

        self._raw_observations = torch.empty(
            (self.batch_size, self.screen_height, self.screen_width, self.num_channels),
            dtype=torch.uint8,
            device=self.device,
        )
        self._rewards = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self._terminated = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._truncated = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._kernel_done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        self._scores = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self._lives = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self._decision_rewards = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        self._decision_terminated = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._decision_truncated = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._decision_timeout = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._decision_raw_frames_advanced = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self._decision_still_active = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        self._raw_action_values = torch.zeros(self.batch_size, dtype=torch.uint8, device=self.device)
        self._packed_decision = torch.zeros(
            (self.batch_size, _DECISION_COLUMN_COUNT),
            dtype=torch.float64,
            device=self.device,
        )

    @property
    def score(self) -> torch.Tensor:
        return self._scores

    @property
    def lives(self) -> torch.Tensor:
        return self._lives

    def observations(self) -> torch.Tensor:
        return self._raw_observations.permute(0, 3, 1, 2).contiguous()

    def reset_batch(
        self,
        seeds: Sequence[int | None] | int | None = None,
        indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        target_indices = self._indices(indices)
        self._set_reset_seeds(seeds, target_indices)
        if indices is None:
            self.env.reset(self.rand_states.data_ptr())
        else:
            index_t = torch.as_tensor(target_indices, dtype=torch.int64, device=self.device).contiguous()
            self.kernels.demon_attack_reset_indices(
                index_t,
                self.states,
                self.frame_states,
                self.ram,
                self.cached_states,
                self.cached_frame_states,
                self.cached_ram,
                self.rand_states,
                self.cache_index,
                int(self.env.state_size()),
                int(self.env.frame_state_size()),
                int(self.rom.ram_size()),
                int(self.noop_max),
            )
        self._rewards.zero_()
        index_t = torch.as_tensor(target_indices, dtype=torch.int64, device=self.device)
        self._terminated[index_t] = False
        self._truncated[index_t] = False
        self._scores[index_t] = 0
        self._kernel_done.fill_(True)
        self._kernel_done[index_t] = False
        self.env.get_data(
            self.terminal_on_life_loss,
            self._kernel_done.data_ptr(),
            self._rewards.data_ptr(),
            self._lives.data_ptr(),
        )
        self._terminated[index_t] = self._kernel_done[index_t]
        self.env.generate_frames(True, True, self.num_channels, self._raw_observations.data_ptr())
        scores = self._scores.detach().cpu().tolist()
        lives = self._lives.detach().cpu().tolist()
        infos = [{"score": int(scores[env_idx]), "ale.lives": int(lives[env_idx])} for env_idx in range(self.batch_size)]
        return self.observations(), infos

    def step_batch(
        self,
        actions: Sequence[int] | np.ndarray | torch.Tensor,
        active_mask: Sequence[bool] | np.ndarray | torch.Tensor | None = None,
    ) -> CudaDemonAttackStepResult:
        action_indices = torch.as_tensor(actions, dtype=torch.int64, device=self.device).reshape(-1)
        action_values = self.action_set[action_indices]
        if active_mask is None:
            self._active_mask.fill_(True)
            self._kernel_done.zero_()
        else:
            self._active_mask.copy_(torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1))
            self._kernel_done.copy_(~self._active_mask)
        self._rewards.zero_()
        self._terminated.zero_()
        self.env.step(False, action_values.data_ptr(), 0, self._kernel_done.data_ptr())
        self.env.get_data(
            self.terminal_on_life_loss,
            self._kernel_done.data_ptr(),
            self._rewards.data_ptr(),
            self._lives.data_ptr(),
        )
        self._terminated.copy_(self._kernel_done & self._active_mask)
        self._scores += self._rewards.to(torch.int32)
        self.env.reset_states()
        self.env.generate_frames(True, True, self.num_channels, self._raw_observations.data_ptr())
        self._truncated.zero_()
        return CudaDemonAttackStepResult(
            observations=self.observations(),
            rewards=self._rewards,
            terminated=self._terminated,
            truncated=self._truncated,
            scores=self._scores,
            lives=self._lives,
        )

    def step_decision(
        self,
        latency: Any,
        actions: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        active_mask: Sequence[bool] | np.ndarray | torch.Tensor,
        episode_step_idx: torch.Tensor,
        max_episode_steps: int,
    ) -> CudaDemonAttackDecisionResult:
        action_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).reshape(-1).contiguous()
        active_t = torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1).contiguous()
        episode_steps = episode_step_idx.view(-1)
        self.env.step_decision(
            bool(self.terminal_on_life_loss),
            int(latency.clock.obs_stride_raw_frames),
            int(self.num_channels),
            int(latency.clock.obs_stride_raw_frames if latency.hold_policy == ActionHoldPolicy.HOLD else 1),
            int(0 if latency.ordering_policy == OrderingPolicy.ISSUE_ORDER_FIFO else 1),
            int(max_episode_steps),
            int(latency.fixed_latency_raw_frames),
            int(latency.latency_distribution_type),
            float(latency.latency_distribution_mean),
            float(latency.latency_distribution_std),
            float(latency.latency_distribution_min),
            float(latency.latency_distribution_max),
            int(self.batch_size),
            int(latency.queue_capacity),
            action_t.data_ptr(),
            active_t.data_ptr(),
            latency.env_step_idx.data_ptr(),
            latency.decision_step_idx.data_ptr(),
            latency.next_action_id.data_ptr(),
            latency.current_latency_raw_frames.data_ptr(),
            latency.current_applied_action.data_ptr(),
            latency.persistence_remaining.data_ptr(),
            latency.pending_actions.data_ptr(),
            latency.pending_ready_raw_frames.data_ptr(),
            latency.pending_action_ids.data_ptr(),
            latency.pending_head.data_ptr(),
            latency.pending_count.data_ptr(),
            latency.scratch_actions.data_ptr(),
            latency.scratch_ready_raw_frames.data_ptr(),
            latency.scratch_action_ids.data_ptr(),
            latency._decision_queued_actions.data_ptr(),
            latency._decision_latency_raw_frames.data_ptr(),
            latency._decision_ready_raw_frames.data_ptr(),
            self._decision_timeout.data_ptr(),
            self._packed_decision.data_ptr(),
            self._decision_rewards.data_ptr(),
            self._decision_terminated.data_ptr(),
            self._decision_truncated.data_ptr(),
            self._decision_raw_frames_advanced.data_ptr(),
            episode_steps.data_ptr(),
            latency.latency_rng_state.data_ptr(),
            self.action_set.data_ptr(),
            self._raw_action_values.data_ptr(),
            self._decision_still_active.data_ptr(),
            self._kernel_done.data_ptr(),
            self._rewards.data_ptr(),
            self._lives.data_ptr(),
            self._scores.data_ptr(),
            self._raw_observations.data_ptr(),
        )
        packed = self._packed_decision.detach().cpu().numpy()
        latency._latency_raw_frames_cpu[:] = packed[:, _DECISION_LATENCY_RAW_FRAMES].astype(np.int64)
        return CudaDemonAttackDecisionResult(
            observations=self.observations(),
            rewards=packed[:, _DECISION_REWARD].astype(np.float32).copy(),
            terminated=packed[:, _DECISION_TERMINATED].astype(np.bool_),
            truncated=packed[:, _DECISION_TRUNCATED].astype(np.bool_),
            scores=packed[:, _DECISION_SCORE].astype(np.int64),
            lives=packed[:, _DECISION_LIVES].astype(np.int64),
            raw_frames_advanced=packed[:, _DECISION_RAW_FRAMES_ADVANCED].astype(np.int64),
            latency_raw_frames=latency._latency_raw_frames_cpu.copy(),
            queued_actions=packed[:, _DECISION_QUEUED_ACTION].astype(np.int64),
            applied_actions=packed[:, _DECISION_APPLIED_ACTION].astype(np.int64),
            ready_raw_frames=packed[:, _DECISION_READY_RAW_FRAME].astype(np.int64),
            pending_counts=packed[:, _DECISION_PENDING_COUNT].astype(np.int64),
            env_step_indices=packed[:, _DECISION_ENV_STEP_IDX].astype(np.int64),
            decision_step_indices=packed[:, _DECISION_DECISION_STEP_IDX].astype(np.int64),
            timeout_mask=packed[:, _DECISION_TIMEOUT].astype(np.bool_),
            rewards_gpu=self._decision_rewards,
            terminated_gpu=self._decision_terminated,
            truncated_gpu=self._decision_truncated,
        )

    def _indices(self, indices: Sequence[int] | np.ndarray | torch.Tensor | None) -> list[int]:
        if indices is None:
            return list(range(self.batch_size))
        if isinstance(indices, torch.Tensor):
            return [int(item) for item in indices.reshape(-1).detach().cpu().tolist()]
        return [int(item) for item in np.asarray(indices).reshape(-1).tolist()]

    def _set_reset_seeds(self, seeds: Sequence[int | None] | int | None, indices: Sequence[int]) -> None:
        index_t = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        if seeds is None:
            self.rand_states[index_t] = torch.randint(
                0,
                np.iinfo(np.int32).max,
                (len(indices),),
                dtype=torch.int32,
                device=self.device,
            )
        elif isinstance(seeds, (int, np.integer)):
            self.rand_states[index_t] = int(seeds)
        else:
            values = [int(seed) for seed in seeds]
            self.rand_states[index_t] = torch.tensor(
                values,
                dtype=torch.int32,
                device=self.device,
            )
