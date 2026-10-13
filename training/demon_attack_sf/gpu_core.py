from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Sequence

import numpy as np
import torch
from gymnasium.utils import seeding

from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
from training.common.utils import batch_indices
from training.demon_attack_sf.cule_atari_extension import load_cule_atari_extension


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
_DECISION_REWARD_CLIPPED = 14
_DECISION_COLUMN_COUNT = 15

_FIRE_RESET_ACTIONS = (1, 2)
_LIFE_LOSS_RESET_ACTIONS = (0, 1, 2)


def load_demon_attack_rom_bytes() -> bytes:
    return resources.files("AutoROM.roms").joinpath("demon_attack.bin").read_bytes()


def demon_attack_rom_path() -> str:
    return str(resources.files("AutoROM.roms").joinpath("demon_attack.bin"))


@dataclass
class _CudaAtariStepResult:
    observations: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    scores: torch.Tensor
    lives: torch.Tensor


@dataclass
class _CudaAtariDecisionResult:
    observations: torch.Tensor
    rewards: np.ndarray
    rewards_clipped: np.ndarray
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
    command_admitted: np.ndarray
    observation_dropped: np.ndarray
    worker_ids: np.ndarray
    worker_service_raw_frames: np.ndarray
    rewards_gpu: torch.Tensor
    rewards_clipped_gpu: torch.Tensor
    terminated_gpu: torch.Tensor
    truncated_gpu: torch.Tensor


@dataclass
class _CudaAtariDecisionGpuResult:
    observations: torch.Tensor
    rewards_gpu: torch.Tensor
    rewards_clipped_gpu: torch.Tensor
    terminated_gpu: torch.Tensor
    truncated_gpu: torch.Tensor
    raw_frames_advanced_gpu: torch.Tensor
    latency_raw_frames_gpu: torch.Tensor
    queued_actions_gpu: torch.Tensor
    applied_actions_gpu: torch.Tensor
    ready_raw_frames_gpu: torch.Tensor
    pending_counts_gpu: torch.Tensor
    env_step_indices_gpu: torch.Tensor
    decision_step_indices_gpu: torch.Tensor
    timeout_mask_gpu: torch.Tensor
    scores_gpu: torch.Tensor
    lives_gpu: torch.Tensor
    command_admitted: np.ndarray
    observation_dropped: np.ndarray
    worker_ids: np.ndarray
    worker_service_raw_frames: np.ndarray


class CudaAtariCore:
    """Run a supported CuLE Atari ROM and decision cadence entirely on CUDA."""

    def __init__(
        self,
        batch_size: int,
        *,
        rom_path: str,
        reset_actions: Sequence[int],
        life_loss_reset_actions: Sequence[int],
        device: str | torch.device = "cuda",
        noop_max: int = 30,
        terminal_on_life_loss: bool = False,
        obs_stride: int = 1,
        frame_stack: int = 1,
    ) -> None:
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.noop_max = int(noop_max)
        self.terminal_on_life_loss = bool(terminal_on_life_loss)
        self.obs_stride = int(obs_stride)
        self.frame_stack = frame_stack
        self.obs_frame_count = min(self.frame_stack, self.obs_stride)
        self.reset_actions = tuple(reset_actions)
        self.life_loss_reset_actions = tuple(life_loss_reset_actions)
        self.ext = load_cule_atari_extension()
        self.rom = self.ext.AtariRom(rom_path)
        self.env = self.ext.AtariEnv(self.rom, self.batch_size, self.noop_max)
        self.native_screen_height = int(self.rom.screen_height())
        self.native_screen_width = int(self.rom.screen_width())
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
        self.cached_frames = torch.zeros(
            (self.noop_max, 300 * self.native_screen_width),
            dtype=torch.uint8,
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
            self.action_set.data_ptr(),
            self.rand_states.data_ptr(),
            self.cached_states.data_ptr(),
            self.cached_ram.data_ptr(),
            self.cached_frame_states.data_ptr(),
            self.cached_frames.data_ptr(),
            self.cache_index.data_ptr(),
        )

        self._obs84_single = torch.empty(
            (self.batch_size, self.screen_height, self.screen_width),
            dtype=torch.uint8,
            device=self.device,
        )
        self._obs84_multi = torch.empty(
            (self.obs_frame_count, self.batch_size, self.screen_height, self.screen_width),
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
        self._decision_rewards_clipped = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
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
        self._native_rgb_frames = torch.empty(
            (self.batch_size, self.native_screen_height, self.native_screen_width, 3),
            dtype=torch.uint8,
            device=self.device,
        )

    @property
    def score(self) -> torch.Tensor:
        return self._scores

    @property
    def lives(self) -> torch.Tensor:
        return self._lives

    @property
    def reset_raw_frame_counts(self) -> torch.Tensor:
        return self.cache_index.to(torch.int64) + 1 + len(self.reset_actions)

    def observations(self) -> torch.Tensor:
        return self._obs84_single.unsqueeze(1).contiguous()

    def native_rgb_frames(self) -> torch.Tensor:
        """Convert the current Atari frame buffer to RGB for each GPU env.

        Repeated calls preserve emulator and render state.

        Returns:
            Tensor with shape ``(batch, native_height, native_width, 3)`` and dtype ``uint8``.
        """
        self.env.generate_frames(False, 3, self._native_rgb_frames.data_ptr())
        return self._native_rgb_frames

    def reset_batch(
        self,
        seeds: Sequence[int | None] | int | None = None,
        indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        target_indices = self._indices(indices)
        reset_cache_indices = self._set_reset_seeds(seeds, target_indices)
        if indices is None:
            self.env.reset(self.rand_states.data_ptr())
            self._reset_indices_from_cache(target_indices, reset_cache_indices)
        else:
            self._reset_indices_from_cache(target_indices, reset_cache_indices)
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
        self._run_reset_action_sequence(target_indices, self.reset_actions)
        self.env.generate_obs84(self._obs84_single.data_ptr())
        reset_obs = self.observations()
        self._rewards.zero_()
        self._terminated[index_t] = False
        self._truncated[index_t] = False
        self._scores[index_t] = 0
        scores = self._scores.detach().cpu().tolist()
        lives = self._lives.detach().cpu().tolist()
        infos = [{"score": int(scores[env_idx]), "ale.lives": int(lives[env_idx])} for env_idx in range(self.batch_size)]
        return reset_obs, infos

    def step_batch(
        self,
        actions: Sequence[int] | np.ndarray | torch.Tensor,
        active_mask: Sequence[bool] | np.ndarray | torch.Tensor | None = None,
    ) -> _CudaAtariStepResult:
        action_indices = torch.as_tensor(actions, dtype=torch.int64, device=self.device).reshape(-1)
        if active_mask is None:
            self._active_mask.fill_(True)
        else:
            self._active_mask.copy_(torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1))
        self._step_active_action_indices(action_indices, self._active_mask)
        return _CudaAtariStepResult(
            observations=self.observations(),
            rewards=self._rewards,
            terminated=self._terminated,
            truncated=self._truncated,
            scores=self._scores,
            lives=self._lives,
        )

    def soft_reset_life_loss_batch(
        self,
        indices: Sequence[int] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        return self._run_reset_action_sequence(self._indices(indices), self.life_loss_reset_actions)

    def step_decision(
        self,
        latency: Any,
        actions: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        active_mask: Sequence[bool] | np.ndarray | torch.Tensor,
        episode_step_idx: torch.Tensor,
        max_episode_steps: int,
    ) -> _CudaAtariDecisionResult:
        gpu_result = self.step_decision_gpu(
            latency,
            actions,
            active_mask=active_mask,
            episode_step_idx=episode_step_idx,
            max_episode_steps=max_episode_steps,
        )
        packed = self._packed_decision.detach().cpu().numpy()
        latency._latency_raw_frames_cpu[:] = packed[:, _DECISION_LATENCY_RAW_FRAMES].astype(np.int64)
        return _CudaAtariDecisionResult(
            observations=gpu_result.observations,
            rewards=packed[:, _DECISION_REWARD].astype(np.float32).copy(),
            rewards_clipped=packed[:, _DECISION_REWARD_CLIPPED].astype(np.float32).copy(),
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
            command_admitted=gpu_result.command_admitted,
            observation_dropped=gpu_result.observation_dropped,
            worker_ids=gpu_result.worker_ids,
            worker_service_raw_frames=gpu_result.worker_service_raw_frames,
            rewards_gpu=gpu_result.rewards_gpu,
            rewards_clipped_gpu=gpu_result.rewards_clipped_gpu,
            terminated_gpu=gpu_result.terminated_gpu,
            truncated_gpu=gpu_result.truncated_gpu,
        )

    def step_decision_gpu(
        self,
        latency: Any,
        actions: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        active_mask: Sequence[bool] | np.ndarray | torch.Tensor,
        episode_step_idx: torch.Tensor,
        max_episode_steps: int,
    ) -> _CudaAtariDecisionGpuResult:
        action_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).reshape(-1).contiguous()
        active_t = torch.as_tensor(active_mask, dtype=torch.bool, device=self.device).reshape(-1).contiguous()
        episode_steps = episode_step_idx.view(-1)
        issue_frames = latency.env_step_idx.clone()
        admitted, worker_ids = latency._prepare_admission(active_t)
        profile_service = latency._sample_profile_latencies(admitted, worker_ids)
        self.env.step_decision(
            bool(self.terminal_on_life_loss),
            int(latency.clock.obs_stride_raw_frames),
            int(self.obs_frame_count),
            int(self.num_channels),
            latency.hold_policy == ActionHoldPolicy.HOLD,
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
            admitted.data_ptr(),
            latency.env_step_idx.data_ptr(),
            latency.decision_step_idx.data_ptr(),
            latency.next_action_id.data_ptr(),
            latency.current_latency_raw_frames.data_ptr(),
            latency.current_applied_action.data_ptr(),
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
            self._decision_rewards_clipped.data_ptr(),
            self._decision_terminated.data_ptr(),
            self._decision_truncated.data_ptr(),
            self._decision_raw_frames_advanced.data_ptr(),
            episode_steps.data_ptr(),
            latency.latency_rng_state.data_ptr(),
            latency._profile_latency_raw_frames.data_ptr(),
            self.action_set.data_ptr(),
            self._raw_action_values.data_ptr(),
            self._decision_still_active.data_ptr(),
            self._kernel_done.data_ptr(),
            self._rewards.data_ptr(),
            self._lives.data_ptr(),
            self._scores.data_ptr(),
            self._obs84_multi.data_ptr(),
        )
        latency._record_admitted_service(admitted, worker_ids, issue_frames, profile_service)
        decision_obs = self._obs84_multi.permute(1, 0, 2, 3).contiguous()
        command_admitted = latency._decision_admitted.detach().cpu().numpy().copy()
        observation_dropped = latency._decision_dropped.detach().cpu().numpy().copy()
        worker_ids_cpu = latency._decision_worker_ids.detach().cpu().numpy().copy()
        worker_service_raw_frames = (
            latency._decision_worker_service_raw_frames.detach().cpu().numpy().copy()
        )
        return _CudaAtariDecisionGpuResult(
            observations=decision_obs,
            rewards_gpu=self._decision_rewards,
            rewards_clipped_gpu=self._decision_rewards_clipped,
            terminated_gpu=self._decision_terminated,
            truncated_gpu=self._decision_truncated,
            raw_frames_advanced_gpu=self._decision_raw_frames_advanced,
            latency_raw_frames_gpu=latency._decision_latency_raw_frames,
            queued_actions_gpu=latency._decision_queued_actions,
            applied_actions_gpu=latency.current_applied_action,
            ready_raw_frames_gpu=latency._decision_ready_raw_frames,
            pending_counts_gpu=latency.pending_count,
            env_step_indices_gpu=latency.env_step_idx,
            decision_step_indices_gpu=latency.decision_step_idx,
            timeout_mask_gpu=self._decision_timeout,
            scores_gpu=self._scores,
            lives_gpu=self._lives,
            command_admitted=command_admitted,
            observation_dropped=observation_dropped,
            worker_ids=worker_ids_cpu,
            worker_service_raw_frames=worker_service_raw_frames,
        )

    def _indices(self, indices: Sequence[int] | np.ndarray | torch.Tensor | None) -> list[int]:
        return batch_indices(indices, batch_size=self.batch_size)

    def _set_reset_seeds(self, seeds: Sequence[int | None] | int | None, indices: Sequence[int]) -> torch.Tensor:
        index_t = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        if seeds is None:
            values_t = torch.randint(
                0,
                np.iinfo(np.int32).max,
                (len(indices),),
                dtype=torch.int32,
                device=self.device,
            )
            self.rand_states[index_t] = values_t
            values = [int(seed) for seed in values_t.detach().cpu().tolist()]
        elif isinstance(seeds, (int, np.integer)):
            self.rand_states[index_t] = int(seeds)
            values = [int(seeds)] * len(indices)
        else:
            values = [int(seed) for seed in seeds]
            self.rand_states[index_t] = torch.tensor(
                values,
                dtype=torch.int32,
                device=self.device,
            )
        cache_indices = [self._cpu_reference_noop_cache_index(seed) for seed in values]
        return torch.tensor(cache_indices, dtype=torch.int64, device=self.device)

    def _cpu_reference_noop_cache_index(self, seed: int) -> int:
        np_seed = int(np.random.SeedSequence(int(seed)).generate_state(n_words=2)[0])
        rng, _ = seeding.np_random(np_seed)
        return int(rng.integers(1, int(self.noop_max) + 1)) - 1

    def _reset_indices_from_cache(self, target_indices: Sequence[int], cache_indices: torch.Tensor) -> None:
        index_t = torch.as_tensor(target_indices, dtype=torch.int64, device=self.device).contiguous()
        self.cache_index[index_t] = cache_indices.to(torch.int32)
        self.states[index_t] = self.cached_states[cache_indices]
        self.frame_states[index_t] = self.cached_frame_states[cache_indices]
        self.ram[index_t] = self.cached_ram[cache_indices]
        self.frame_buffer[index_t] = self.cached_frames[cache_indices]

    def _run_reset_action_sequence(self, indices: Sequence[int], actions: Sequence[int]) -> torch.Tensor:
        index_t = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
        active = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        active[index_t] = True
        action_t = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        obs = self.observations()
        for action in actions:
            action_t[index_t] = int(action)
            obs = self._step_active_action_indices(action_t, active)
        return obs

    def _step_active_action_indices(self, action_indices: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        action_values = self.action_set[action_indices]
        self._kernel_done.copy_(~active)
        self._rewards.zero_()
        self._terminated.zero_()
        self.env.step(False, action_values.data_ptr(), 0, self._kernel_done.data_ptr())
        self.env.get_data(
            self.terminal_on_life_loss,
            self._kernel_done.data_ptr(),
            self._rewards.data_ptr(),
            self._lives.data_ptr(),
        )
        self._terminated.copy_(self._kernel_done & active)
        self._scores += self._rewards.to(torch.int32)
        self.env.generate_obs84(self._obs84_single.data_ptr())
        self._truncated.zero_()
        return self.observations()


class CudaDemonAttackCore(CudaAtariCore):
    """Configure the shared CUDA Atari core for Demon Attack reset semantics."""

    def __init__(
        self,
        batch_size: int,
        *,
        device: str | torch.device = "cuda",
        noop_max: int = 30,
        terminal_on_life_loss: bool = False,
        obs_stride: int = 1,
        frame_stack: int = 1,
    ) -> None:
        super().__init__(
            batch_size,
            rom_path=demon_attack_rom_path(),
            reset_actions=_FIRE_RESET_ACTIONS,
            life_loss_reset_actions=_LIFE_LOSS_RESET_ACTIONS,
            device=device,
            noop_max=noop_max,
            terminal_on_life_loss=terminal_on_life_loss,
            obs_stride=obs_stride,
            frame_stack=frame_stack,
        )
