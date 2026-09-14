"""Batched training transport with the evaluator's action scheduling semantics."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import derive_seed
from latency_bench.core.types import Action, ActionHoldPolicy, OrderingPolicy
from latency_bench.latency.samplers import build_episode_sampler_factory, build_sampler
from latency_bench.scheduler.factory import build_action_scheduler


class ActionLatencyBatch:
    """Batch transport clocks and queues on CPU; retain action payloads on device.

    Each pending entry represents an issued tensor batch, with independent row
    readiness. Sampling keeps the evaluator's per-replica random streams.
    """

    def __init__(self, config, *, num_envs, device, noop_command):
        self.num_envs = num_envs
        self.device = device
        self.clock = EnvClock(env_fps=config["env"]["env_fps"], obs_fps=config["env"]["obs_fps"])
        self.noop_command = torch.as_tensor(noop_command, device=device, dtype=torch.float32)
        scheduler = build_action_scheduler(
            config["scheduler"], clock=self.clock,
            noop_action=Action(self.noop_command, name="noop", is_noop=True),
        )
        self.ordering_policy = scheduler.ordering_policy
        self.hold = ActionHoldPolicy(config["scheduler"]["hold_policy"]) == ActionHoldPolicy.HOLD
        self.hold_last_chunk_action = scheduler.hold_last_chunk_action
        self.capacity = config["executor"]["simulated_worker_capacity"]
        self._all_ids = np.arange(num_envs)
        self._frames = np.zeros(num_envs, dtype=np.int64)
        self.episodes = [-1] * num_envs
        self._submitted = np.zeros(num_envs, dtype=np.int64)
        self._dropped = np.zeros(num_envs, dtype=np.int64)
        self._last_dropped = np.zeros(num_envs, dtype=bool)
        self.last_submission = [None] * num_envs
        self._busy_until = np.zeros((num_envs, 1 if self.capacity is None else self.capacity), dtype=np.int64)
        self._current_batch = np.full(num_envs, -1, dtype=np.int64)
        self._chunk_position = np.zeros(num_envs, dtype=np.int64)
        self._application = np.full((num_envs, 3), -1, dtype=np.int64)
        self._application_seen = np.zeros(num_envs, dtype=bool)
        self._batches = {}
        self._next_batch = 0
        self.samplers = []
        self.sampler_factory = build_episode_sampler_factory(config["latency"], self.clock)
        for env_id in range(num_envs):
            if self.sampler_factory is None:
                latency = dict(config["latency"])
                latency["seed"] = derive_seed(latency["seed"], vector_index=env_id)
                sampler = build_sampler(latency, clock=self.clock)
            else:
                sampler = self.sampler_factory(env_id)
            self.samplers.append(sampler)

    @property
    def frames(self):
        return self._frames.tolist()

    @property
    def submitted(self):
        return self._submitted.tolist()

    @property
    def dropped(self):
        return self._dropped.tolist()

    @property
    def last_dropped(self):
        return self._last_dropped.tolist()

    @property
    def last_application(self):
        return [
            dict(zip(("source_raw_frame", "source_obs_id", "chunk_index"), values)) if seen else None
            for seen, values in zip(self._application_seen.tolist(), self._application.tolist())
        ]

    def _ids(self, env_ids):
        return self._all_ids if env_ids is None else np.asarray(env_ids, dtype=np.int64)

    def _clear(self, ids):
        self._frames[ids] = 0
        self._submitted[ids] = 0
        self._dropped[ids] = 0
        self._last_dropped[ids] = False
        self._busy_until[ids] = 0
        self._current_batch[ids] = -1
        self._chunk_position[ids] = 0
        self._application[ids] = -1
        self._application_seen[ids] = False
        for env_id in ids:
            self.last_submission[env_id] = None
        for batch in self._batches.values():
            batch["pending"][ids] = False
        self._discard_finished_batches()

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        next_episode_id = max(self.episodes) + 1
        self._clear(ids)
        for env_id in ids:
            self.episodes[env_id] = next_episode_id
            next_episode_id += 1
            if self.sampler_factory is not None:
                self.samplers[env_id] = self.sampler_factory(self.episodes[env_id])

    def submit(self, commands, env_ids=None):
        ids = self._ids(env_ids)
        self._last_dropped[ids] = False
        for env_id in ids:
            self.last_submission[env_id] = None
        due = ids[self._frames[ids] % self.clock.obs_stride_raw_frames == 0]
        idle = self._busy_until[due] <= self._frames[due, None]
        if self.capacity is None and (~idle.any(axis=1)).any():
            self._busy_until = np.pad(self._busy_until, ((0, 0), (0, 1)))
            idle = self._busy_until[due] <= self._frames[due, None]
        available = idle.any(axis=1)
        rejected = due[~available]
        self._last_dropped[rejected] = True
        self._dropped[rejected] += 1
        accepted = due[available]
        admitted = np.zeros(self.num_envs, dtype=bool)
        admitted[accepted] = True
        if not accepted.size:
            return torch.as_tensor(admitted, device=self.device)
        workers = idle[available].argmax(axis=1)
        in_flight = (~idle[available]).sum(axis=1)
        chunked = commands.ndim == 3
        horizon = commands.shape[1] if chunked else 1
        batch = {
            "commands": commands.detach().clone().reshape(-1, self.noop_command.numel()),
            "horizon": horizon, "chunked": chunked,
            "source": self._frames.copy(), "obs_id": self._submitted.copy(),
            "ready": np.zeros(self.num_envs, dtype=np.int64), "pending": admitted.copy(),
        }
        # Convert batch metadata once at the scalar sampler/JSON boundary.
        for env_id, worker, count, frame, obs_id, drops in zip(
            accepted.tolist(), workers.tolist(), in_flight.tolist(),
            self._frames[accepted].tolist(), self._submitted[accepted].tolist(),
            self._dropped[accepted].tolist(),
        ):
            sample = self.samplers[env_id].sample({
                "episode_id": self.episodes[env_id], "worker_slot": worker,
                "slot_id": env_id, "obs_id": obs_id, "raw_frame": frame,
                "env_step": frame, "sim_time_ms": self.clock.step_to_time_ms(frame),
                "in_flight_count": count,
                "idle_worker_count": None if self.capacity is None else self.capacity - count,
                "recent_drop_count": drops,
            })
            ready = frame + sample.action_ready_raw_frames
            batch["ready"][env_id] = ready
            self._busy_until[env_id, worker] = frame + sample.worker_service_raw_frames
            self.last_submission[env_id] = {
                "episode_id": self.episodes[env_id], "obs_id": obs_id,
                "issued_raw_frame": frame, "ready_raw_frame": ready,
                "latency_ms": sample.latency_ms, "worker_id": worker,
            }
        self._submitted[accepted] += 1
        self._batches[self._next_batch] = batch
        self._next_batch += 1
        return torch.as_tensor(admitted, device=self.device)

    def _select(self):
        selected = np.full(self.num_envs, -1, dtype=np.int64)
        blocked = np.zeros(self.num_envs, dtype=bool)
        for batch_id, batch in self._batches.items():
            ready = batch["pending"] & (batch["ready"] <= self._frames)
            if self.ordering_policy == OrderingPolicy.ISSUE_ORDER_FIFO:
                ready &= ~blocked
                blocked |= batch["pending"]
            selected[ready] = batch_id
        return selected

    def actions(self, env_ids=None):
        ids = self._ids(env_ids)
        active = np.zeros(self.num_envs, dtype=bool)
        active[ids] = True
        selected = self._select()
        selected = np.where(active, selected, -1)
        replacing = selected >= 0
        self._current_batch[replacing] = selected[replacing]
        self._application_seen[ids] = True
        self._application[ids] = -1
        result = np.zeros(self.num_envs, dtype=np.int64)
        banks = [self.noop_command.reshape(1, -1)]
        bank_rows = 1
        for batch_id, batch in self._batches.items():
            chosen = selected == batch_id
            if self.ordering_policy == OrderingPolicy.LATEST_READY:
                batch["pending"] &= ~((selected >= batch_id) & replacing)
            else:
                batch["pending"][chosen] = False
            self._chunk_position[chosen] = self._frames[chosen] - batch["source"][chosen]
            current = active & (self._current_batch == batch_id)
            horizon = batch["horizon"]
            if batch["chunked"]:
                position = np.minimum(self._chunk_position, horizon - 1)
                applying = current & ((self._chunk_position < horizon) | self.hold_last_chunk_action)
                self._chunk_position[current & (self._chunk_position < horizon)] += 1
                self._current_batch[current & ~applying] = -1
            else:
                position = np.zeros(self.num_envs, dtype=np.int64)
                applying = current
                if not self.hold:
                    self._current_batch[current] = -1
            rows = self._all_ids[applying]
            if rows.size:
                banks.append(batch["commands"])
                result[rows] = bank_rows + rows * horizon + position[rows]
                bank_rows += batch["commands"].shape[0]
                self._application[rows, 0] = batch["source"][rows]
                self._application[rows, 1] = batch["obs_id"][rows]
                self._application[rows, 2] = np.minimum(self._frames[rows] - batch["source"][rows], horizon - 1)
        self._discard_finished_batches()
        commands = torch.cat(banks)
        return commands.index_select(0, torch.as_tensor(result, device=self.device)).reshape(
            self.num_envs, *self.noop_command.shape
        ).to(self.noop_command)

    def _discard_finished_batches(self):
        current = set(np.unique(self._current_batch))
        self._batches = {
            batch_id: batch for batch_id, batch in self._batches.items()
            if batch["pending"].any() or batch_id in current
        }

    def advance(self, env_ids=None):
        self._frames[self._ids(env_ids)] += 1

    def state_dict(self):
        return {"samplers": deepcopy(self.samplers), "episodes": list(self.episodes)}

    def load_state_dict(self, state):
        self.samplers = deepcopy(state["samplers"])
        self.episodes = list(state["episodes"])
        self._clear(self._all_ids)


class TorchActionLatencyWrapper:
    """Apply action latency outside a native tensor vector env's autoreset."""

    def __init__(self, env, config, *, num_envs, device, noop_command):
        self.env = env
        self.latency = ActionLatencyBatch(
            config, num_envs=num_envs, device=device, noop_command=noop_command
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.latency.reset()
        return observation, info

    def step(self, action):
        issued = action.clamp(-1, 1)
        admitted = self.latency.submit(issued)
        dropped = torch.tensor(self.latency.last_dropped, device=issued.device)
        applied = self.latency.actions()
        observation, reward, terminated, truncated, info = self.env.step(applied)
        self.latency.advance()
        info["issued_command"] = issued
        info["applied_command"] = applied
        info["command_admitted"] = admitted
        info["command_dropped"] = dropped
        # The native vector wrapper has already preserved terminal observations
        # and reset these slots. Only transport state is reset here.
        self.latency.reset(torch.nonzero(terminated | truncated).flatten().tolist())
        return observation, reward, terminated, truncated, info

    def close(self):
        return self.env.close()
