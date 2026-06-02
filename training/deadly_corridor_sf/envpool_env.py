from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict as GymDict

from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import (
    derive_latency_seed,
    distribution_max_frame_latency,
    latency_distribution_label,
    make_latency_distribution_sampler,
)
from latency_bench.core.types import Action, ActionEvent, ActionHoldPolicy, LatencyRecord, OrderingPolicy
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.persistence import HoldPolicy, OneFrameThenNoopPolicy
from training.common.batched_metrics import BatchedEpisodeMetrics
from training.common.latency_config import (
    sample_factory_latency_params_from_flat_config,
    sample_factory_latency_seed_from_flat_config,
)
from training.common.latency_wrapper import clone_action


POLICY_HEIGHT = 72
POLICY_WIDTH = 128


def tuple_action_to_envpool_buttons(action: Any) -> np.ndarray:
    turn, move, strafe, attack = action
    buttons = np.zeros(7, dtype=np.uint8)
    if turn:
        buttons[int(turn) - 1] = 1
    if move:
        buttons[2 + int(move) - 1] = 1
    if strafe:
        buttons[4 + int(strafe) - 1] = 1
    if attack:
        buttons[6] = 1
    return buttons


def tuple_action_to_joint_id(action: Any) -> int:
    turn, move, strafe, attack = action
    return int(turn) + 3 * int(move) + 9 * int(strafe) + 27 * int(attack)


class DeadlyCorridorEnvPoolEnv(gym.Env):
    is_multiagent = True

    def __init__(
        self,
        *,
        cfg: Any,
        env_config: Any,
        spec: Any,
        envpool_batch_size: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.env_config = env_config
        self.spec = spec
        self.num_agents = int(envpool_batch_size)
        self.clock = EnvClock(env_fps=cfg.env_fps, obs_fps=cfg.obs_fps)
        self.frame_stack = int(cfg.frame_stack)
        self.raw_width = 256 if bool(cfg.wide_aspect_ratio) else 160
        self.raw_height = 144 if bool(cfg.wide_aspect_ratio) else 120
        self.policy_width = int(cfg.res_w)
        self.policy_height = int(cfg.res_h)
        self.action_space = spec.action_space
        self.observation_space = GymDict(
            {
                "obs": Box(
                    low=0,
                    high=255,
                    shape=(3 * self.frame_stack, self.policy_height, self.policy_width),
                    dtype=np.uint8,
                )
            }
        )
        self.noop_action = (0, 0, 0, 0)
        self._reward_scale = float(spec.reward_scaling)
        self._env_seed = self._sub_seed(0)
        self._env = self._make_env(seed=self._env_seed)
        self._stack = np.empty(
            (self.num_agents, 3 * self.frame_stack, self.policy_height, self.policy_width),
            dtype=np.uint8,
        )
        self._latest_processed_obs = np.empty(
            (self.num_agents, 3, self.policy_height, self.policy_width),
            dtype=np.uint8,
        )
        self._has_reset = False
        self._env_step_idx = np.zeros(self.num_agents, dtype=np.int64)
        self._decision_step_idx = np.zeros(self.num_agents, dtype=np.int64)
        self._episode_idx = np.zeros(self.num_agents, dtype=np.int64)
        self._next_action_id = np.zeros(self.num_agents, dtype=np.int64)
        self._current_latency_raw_frames = np.zeros(self.num_agents, dtype=np.int64)
        self._current_applied_actions: list[Any] = [clone_action(self.noop_action) for _ in range(self.num_agents)]
        self._schedulers = [self._make_scheduler() for _ in range(self.num_agents)]
        self._latency_samplers: list[Any] = [None for _ in range(self.num_agents)]
        self._last_info: list[dict[str, Any]] = [{} for _ in range(self.num_agents)]
        self.sample_factory_step_cpu_tensors: dict[str, Any] = {}

        sub_seeds = [self._sub_seed(agent_idx) for agent_idx in range(self.num_agents)]
        self._latency_seeds = [
            sample_factory_latency_seed_from_flat_config(
                self.cfg,
                fallback_seed=sub_seeds[agent_idx],
                worker_index=self._worker_index,
                vector_index=self._vector_index,
                env_id=self._sub_env_id(agent_idx),
            )
            for agent_idx in range(self.num_agents)
        ]
        self._latency_params = sample_factory_latency_params_from_flat_config(self.cfg, env_fps=self.clock.env_fps)
        self._latency_label = (
            None
            if self._latency_params.latency_distribution_spec is None
            else latency_distribution_label(self._latency_params.latency_distribution_spec)
        )
        self._max_frame_latency = (
            int(self._latency_params.fixed_latency_raw_frames)
            if self._latency_params.latency_distribution_spec is None
            else distribution_max_frame_latency(self._latency_params.latency_distribution_spec)
        )
        self._max_pending_actions = self.cfg.max_pending_actions
        self.metrics = BatchedEpisodeMetrics(
            batch_size=self.num_agents,
            frames_per_decision=self.clock.obs_stride_raw_frames,
            metrics_path=self.cfg.episode_metrics_path,
            worker_index=self._worker_index,
            vector_index=self._vector_index,
            env_ids=[self._sub_env_id(agent_idx) for agent_idx in range(self.num_agents)],
            include_score=False,
        )

    @property
    def _worker_index(self) -> int:
        return int(self.env_config.worker_index)

    @property
    def _vector_index(self) -> int:
        return int(self.env_config.vector_index)

    @property
    def _env_id(self) -> int:
        return int(self.env_config.env_id)

    def _sub_seed(self, agent_idx: int) -> int:
        return int(self.cfg.seed) + 100000 * self._worker_index + 1000 * self._vector_index + int(agent_idx)

    def _sub_env_id(self, agent_idx: int) -> int:
        return self._env_id * self.num_agents + int(agent_idx)

    def _make_env(self, *, seed: int):
        import envpool
        import vizdoom
        from sf_examples.vizdoom.doom import doom_gym

        scenarios_dir = Path(doom_gym.__file__).resolve().parent / "scenarios"
        cfg_path = self._patched_envpool_cfg_path(scenarios_dir / self.spec.env_spec_file)
        wad_path = scenarios_dir / str(self.spec.env_spec_file).replace(".cfg", ".wad")
        vizdoom_dir = Path(vizdoom.__file__).resolve().parent
        return envpool.make_gymnasium(
            "VizdoomCustom-v1",
            cfg_path=str(cfg_path),
            wad_path=str(wad_path),
            vzd_path=str(vizdoom_dir / "vizdoom"),
            iwad_path=str(vizdoom_dir / "freedoom2.wad"),
            num_envs=self.num_agents,
            batch_size=self.num_agents,
            seed=int(seed),
            img_height=self.raw_height,
            img_width=self.raw_width,
            stack_num=1,
            frame_skip=1,
            use_inter_area_resize=False,
            use_native_reward=True,
            preserve_config_screen_resolution=True,
            reward_config={},
            selected_weapon_reward_config={},
            max_episode_steps=2100,
        )

    def _patched_envpool_cfg_path(self, source_path: Path) -> Path:
        target_resolution = f"RES_{self.raw_width}X{self.raw_height}"
        text = source_path.read_text(encoding="utf-8")
        lines = []
        for line in text.splitlines():
            if line.strip().lower().startswith("screen_resolution"):
                prefix = line.split("=", 1)[0].rstrip()
                lines.append(f"{prefix} = {target_resolution}")
            else:
                lines.append(line)
        out_dir = Path(tempfile.gettempdir()) / "latency_bench_envpool"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{source_path.stem}_{target_resolution.lower()}.cfg"
        tmp_file = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=out_dir,
            prefix=f".{out_path.name}.",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = Path(tmp_file.name)
        try:
            with tmp_file as handle:
                handle.write("\n".join(lines) + "\n")
            os.replace(tmp_path, out_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return out_path

    def _make_scheduler(self) -> ActionScheduler:
        return ActionScheduler(
            clock=self.clock,
            noop_action=Action(value=clone_action(self.noop_action), name="noop", is_noop=True),
            persistence_policy=self._make_persistence_policy(),
            ordering_policy=self._ordering_policy(),
        )

    def _make_persistence_policy(self):
        hold_policy = self._hold_policy()
        if hold_policy == ActionHoldPolicy.ONE_FRAME_THEN_NOOP:
            return OneFrameThenNoopPolicy()
        if hold_policy == ActionHoldPolicy.HOLD:
            return HoldPolicy(duration_raw_frames=self.clock.obs_stride_raw_frames)
        raise ValueError(f"Unsupported hold policy: {hold_policy}")

    def _hold_policy(self) -> ActionHoldPolicy:
        return ActionHoldPolicy(str(self.cfg.hold_policy).strip().lower())

    def _ordering_policy(self) -> OrderingPolicy:
        return OrderingPolicy(str(self.cfg.ordering_policy).strip().lower())

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._env.close()
            self._env_seed = int(seed)
            self._env = self._make_env(seed=self._env_seed)
        raw_obs, _info = self._env.reset()
        self.metrics.reset()
        self._reset_latency_state(range(self.num_agents))
        self._env_step_idx[:] = 0
        self._decision_step_idx[:] = 0
        processed = self._process_raw_obs(raw_obs)
        self._latest_processed_obs[:] = processed
        self._reset_stack(processed)
        reset_infos = self._reset_infos(range(self.num_agents))
        self._last_info = [dict(reset_infos[agent_idx]) for agent_idx in range(self.num_agents)]
        self._has_reset = True
        return self._obs(), reset_infos

    def step(self, action):
        if not self._has_reset:
            self.reset()

        actions = self._normalize_actions(action)
        issued_actions = np.asarray([tuple_action_to_joint_id(item) for item in actions], dtype=np.int64)
        rewards = np.zeros(self.num_agents, dtype=np.float64)
        raw_rewards = np.zeros(self.num_agents, dtype=np.float64)
        terminated = np.zeros(self.num_agents, dtype=np.bool_)
        truncated = np.zeros(self.num_agents, dtype=np.bool_)
        raw_frames_advanced = np.zeros(self.num_agents, dtype=np.int64)
        active = np.ones(self.num_agents, dtype=np.bool_)
        latency_raw_frames = np.zeros(self.num_agents, dtype=np.int64)
        ready_raw_frames = np.zeros(self.num_agents, dtype=np.int64)
        pending_counts = np.zeros(self.num_agents, dtype=np.int64)

        for agent_idx, agent_action in enumerate(actions):
            latency_raw_frames[agent_idx] = self._next_latency_raw_frames(agent_idx)
            ready_raw_frames[agent_idx] = self._enqueue_action(agent_idx, agent_action, int(latency_raw_frames[agent_idx]))

        latest_info = [dict(item) for item in self._last_info]
        for _raw_frame in range(self.clock.obs_stride_raw_frames):
            buttons = np.zeros((self.num_agents, 7), dtype=np.float64)
            active_indices = np.flatnonzero(active)
            for agent_idx in active_indices:
                active_action = self._schedulers[int(agent_idx)].update(
                    current_step=int(self._env_step_idx[int(agent_idx)]),
                    current_time_ms=self.clock.step_to_time_ms(int(self._env_step_idx[int(agent_idx)])),
                )
                self._current_applied_actions[int(agent_idx)] = clone_action(active_action.value)
                buttons[int(agent_idx)] = tuple_action_to_envpool_buttons(active_action.value)

            raw_obs, step_rewards, step_terminated, step_truncated, step_info = self._env.step(buttons)
            processed = self._process_raw_obs(raw_obs)
            self._latest_processed_obs[active] = processed[active]
            self._append_stack(processed, active)
            step_rewards = np.asarray(step_rewards, dtype=np.float64).reshape(-1)
            step_terminated = np.asarray(step_terminated, dtype=np.bool_).reshape(-1)
            step_truncated = np.asarray(step_truncated, dtype=np.bool_).reshape(-1)
            split_info = self._split_infos(step_info)

            scaled_rewards = step_rewards * self._reward_scale
            rewards[active] += scaled_rewards[active]
            raw_rewards[active] += scaled_rewards[active] / self._reward_scale
            terminated |= active & step_terminated
            truncated |= active & step_truncated
            raw_frames_advanced[active] += 1
            self._env_step_idx[active] += 1
            for agent_idx in active_indices:
                latest_info[int(agent_idx)] = split_info[int(agent_idx)]

            active &= ~np.logical_or(terminated, truncated)
            if not np.any(active):
                break

        for agent_idx in range(self.num_agents):
            pending_counts[agent_idx] = len(self._schedulers[agent_idx].get_pending_events())

        done_infos = self.metrics.update(
            issued_actions=issued_actions,
            rewards=raw_rewards,
            terminated=terminated,
            truncated=truncated,
            scores=np.zeros(self.num_agents, dtype=np.int64),
            active_mask=np.ones(self.num_agents, dtype=np.bool_),
            latency_raw_frames=latency_raw_frames,
        )
        done_mask = np.logical_or(terminated, truncated)

        done_indices = np.flatnonzero(done_mask)
        infos = self._step_infos(
            latest_info=latest_info,
            raw_rewards=raw_rewards,
            latency_raw_frames=latency_raw_frames,
            ready_raw_frames=ready_raw_frames,
            raw_frames_advanced=raw_frames_advanced,
            pending_counts=pending_counts,
            queued_actions=actions,
            done_infos=done_infos,
        )
        obs = self._obs()
        self._decision_step_idx += 1
        reset_infos = self._auto_reset_done_agents(done_indices, latest_info=latest_info)
        for agent_idx, reset_info in reset_infos.items():
            infos[int(agent_idx)]["reset_info"] = reset_info
        self._last_info = [dict(item) for item in infos]
        self.sample_factory_step_cpu_tensors = {}
        return obs, rewards, terminated, truncated, infos

    def _auto_reset_done_agents(
        self,
        done_indices: Sequence[int],
        *,
        latest_info: Sequence[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        reset_indices = [int(agent_idx) for agent_idx in done_indices]
        if not reset_indices:
            return {}
        raw_obs = np.stack(
            [
                np.asarray(latest_info[agent_idx]["terminal_reset_obs"], dtype=np.uint8)
                for agent_idx in reset_indices
            ],
            axis=0,
        )
        processed = self._process_raw_obs(raw_obs)
        self.metrics.reset(indices=reset_indices)
        self._reset_latency_state(reset_indices)
        self._env_step_idx[reset_indices] = 0
        self._decision_step_idx[reset_indices] = 0
        reset_latency_infos = self._reset_infos(reset_indices)
        reset_infos: dict[int, dict[str, Any]] = {}
        for offset, agent_idx in enumerate(reset_indices):
            self._latest_processed_obs[agent_idx] = processed[offset]
            self._reset_stack_for_agent(agent_idx, processed[offset])
            reset_info = dict(reset_latency_infos[agent_idx])
            reset_infos[agent_idx] = reset_info
            self._last_info[agent_idx] = dict(reset_info)
        return reset_infos

    def _reset_latency_state(self, indices: Sequence[int]) -> None:
        for agent_idx in indices:
            idx = int(agent_idx)
            self._schedulers[idx].reset()
            self._current_applied_actions[idx] = clone_action(self.noop_action)
            self._next_action_id[idx] = 0
            self._current_latency_raw_frames[idx] = (
                0
                if self._latency_params.latency_distribution_spec is not None
                else int(self._latency_params.fixed_latency_raw_frames)
            )
            self._reset_latency_sampler(idx)
            self._episode_idx[idx] += 1

    def _reset_latency_sampler(self, agent_idx: int) -> None:
        if self._latency_params.latency_distribution_spec is None:
            self._latency_samplers[agent_idx] = None
            return
        seed = derive_latency_seed(self._latency_seeds[agent_idx], episode_idx=int(self._episode_idx[agent_idx]))
        self._latency_samplers[agent_idx] = make_latency_distribution_sampler(
            self._latency_params.latency_distribution_spec,
            seed=seed,
        )

    def _next_latency_raw_frames(self, agent_idx: int) -> int:
        if self._latency_params.latency_distribution_spec is None:
            return int(self._latency_params.fixed_latency_raw_frames)
        return int(self._latency_samplers[agent_idx].sample_frame())

    def _enqueue_action(self, agent_idx: int, action: Any, latency_raw_frames: int) -> int:
        obs_env_step = int(self._env_step_idx[agent_idx])
        ready_raw_frame = obs_env_step + int(latency_raw_frames)
        event = ActionEvent(
            action_id=int(self._next_action_id[agent_idx]),
            action=Action(value=clone_action(action), name=str(action), is_noop=False),
            obs_env_step=obs_env_step,
            ready_time_ms=self.clock.step_to_time_ms(ready_raw_frame),
            ready_step_float=float(ready_raw_frame),
            latency=LatencyRecord(
                obs_id=int(self._decision_step_idx[agent_idx]),
                latency_ms=float(latency_raw_frames) * self.clock.frame_ms,
            ),
            latency_raw_frames=int(latency_raw_frames),
            ready_raw_frame=ready_raw_frame,
        )
        self._next_action_id[agent_idx] += 1
        self._schedulers[agent_idx].enqueue(event)
        if self._max_pending_actions is not None and len(self._schedulers[agent_idx].get_pending_events()) > int(
            self._max_pending_actions
        ):
            raise RuntimeError(
                "pending action queue exceeded latency.max_pending_actions="
                f"{self._max_pending_actions}"
            )
        return int(ready_raw_frame)

    def _reset_infos(self, indices: Sequence[int]) -> dict[int, dict[str, Any]]:
        return {
            int(agent_idx): self._latency_info(
                int(agent_idx),
                int(self._current_latency_raw_frames[int(agent_idx)]),
                episode_idx=int(self._episode_idx[int(agent_idx)] - 1),
            )
            for agent_idx in indices
        }

    def _latency_info(self, agent_idx: int, latency_raw_frames: int, *, episode_idx: int) -> dict[str, Any]:
        if not bool(self.cfg.add_latency_info):
            return {}
        info = {
            "episode_idx": int(episode_idx),
            "decision_step_idx": int(self._decision_step_idx[agent_idx]),
            "env_step_idx": int(self._env_step_idx[agent_idx]),
            "raw_frame": int(self._env_step_idx[agent_idx]),
            "applied_action": clone_action(self._current_applied_actions[agent_idx]),
            "pending_action_count": int(len(self._schedulers[agent_idx].get_pending_events())),
            "latency_raw_frames": int(latency_raw_frames),
            "latency_ms": float(latency_raw_frames) * self.clock.frame_ms,
            "env_fps": float(self.clock.env_fps),
            "obs_fps": float(self.clock.obs_fps),
            "obs_stride_raw_frames": int(self.clock.obs_stride_raw_frames),
            "hold_policy": self._hold_policy().value,
            "ordering_policy": self._ordering_policy().value,
        }
        if self._latency_params.latency_distribution_spec is not None:
            info["latency_distribution"] = self._latency_params.latency_distribution_spec.to_dict()
            info["latency_distribution_label"] = str(self._latency_label)
        return info

    def _step_infos(
        self,
        *,
        latest_info: Sequence[dict[str, Any]],
        raw_rewards: np.ndarray,
        latency_raw_frames: np.ndarray,
        ready_raw_frames: np.ndarray,
        raw_frames_advanced: np.ndarray,
        pending_counts: np.ndarray,
        queued_actions: Sequence[Any],
        done_infos: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        for agent_idx in range(self.num_agents):
            info = {
                "num_frames": 1,
                "pos": {"agent_x": float("nan"), "agent_y": float("nan"), "agent_a": float("nan")},
                "HEALTH": float(latest_info[agent_idx]["HEALTH"]),
            }
            info["raw_reward"] = float(raw_rewards[agent_idx])
            if bool(self.cfg.add_latency_info):
                info.update(
                    self._latency_info(
                        agent_idx,
                        int(latency_raw_frames[agent_idx]),
                        episode_idx=int(self._episode_idx[agent_idx]),
                    )
                )
                info["queued_action"] = clone_action(queued_actions[agent_idx])
                info["ready_raw_frame"] = int(ready_raw_frames[agent_idx])
                info["raw_frames_advanced"] = int(raw_frames_advanced[agent_idx])
                info["pending_action_count"] = int(pending_counts[agent_idx])
            if agent_idx in done_infos:
                done_info = dict(done_infos[agent_idx])
                if "raw_reward" in done_info:
                    del done_info["raw_reward"]
                info.update(done_info)
            infos.append(info)
        return infos

    def _process_raw_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(raw_obs, dtype=np.uint8)
        out = np.empty((obs.shape[0], 3, self.policy_height, self.policy_width), dtype=np.uint8)
        for row_idx in range(obs.shape[0]):
            hwc = np.transpose(obs[row_idx], (1, 2, 0))
            resized = cv2.resize(hwc, (self.policy_width, self.policy_height), interpolation=cv2.INTER_NEAREST)
            out[row_idx] = np.transpose(resized, (2, 0, 1))
        return out

    def _reset_stack(self, obs: np.ndarray) -> None:
        for agent_idx in range(self.num_agents):
            self._reset_stack_for_agent(agent_idx, obs[agent_idx])

    def _reset_stack_for_agent(self, agent_idx: int, obs: np.ndarray) -> None:
        for frame_idx in range(self.frame_stack):
            start = frame_idx * 3
            self._stack[agent_idx, start : start + 3] = obs

    def _append_stack(self, obs: np.ndarray, active: np.ndarray) -> None:
        rows = np.flatnonzero(active)
        if rows.size == 0:
            return
        if self.frame_stack == 1:
            self._stack[rows] = obs[rows]
        else:
            self._stack[rows, :-3] = self._stack[rows, 3:]
            self._stack[rows, -3:] = obs[rows]

    def _obs(self) -> dict[str, np.ndarray]:
        return {"obs": self._stack.copy()}

    def _split_infos(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        infos = [dict() for _ in range(self.num_agents)]
        for key, value in info.items():
            if isinstance(value, dict):
                continue
            arr = np.asarray(value)
            for agent_idx in range(arr.shape[0]):
                item = arr[agent_idx]
                infos[agent_idx][key] = item.item() if np.asarray(item).shape == () else item
        return infos

    def _normalize_actions(self, action: Any) -> list[tuple[int, int, int, int]]:
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        if isinstance(action, np.ndarray):
            arr = action
            if arr.ndim == 1 and arr.shape[0] == 4 and self.num_agents == 1:
                return [tuple(int(item) for item in arr.tolist())]
            if arr.ndim == 2 and arr.shape[1] == 4:
                return [tuple(int(item) for item in row) for row in arr.tolist()]
        if isinstance(action, tuple) and len(action) == 4:
            branches = []
            for branch in action:
                if hasattr(branch, "detach"):
                    branch = branch.detach().cpu().numpy()
                arr = np.asarray(branch).reshape(-1)
                branches.append(arr)
            return [
                (
                    int(branches[0][agent_idx]),
                    int(branches[1][agent_idx]),
                    int(branches[2][agent_idx]),
                    int(branches[3][agent_idx]),
                )
                for agent_idx in range(self.num_agents)
            ]
        return [tuple(int(item) for item in agent_action) for agent_action in action]

    def close(self):
        self._env.close()
        super().close()
