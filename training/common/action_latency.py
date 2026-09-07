"""Training-side action latency using the evaluator's scheduling primitives."""

from __future__ import annotations

from copy import deepcopy

import torch

from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import derive_seed
from latency_bench.core.types import Action, ActionEvent, LatencyRecord, OrderingPolicy
from latency_bench.executors._simulated_timeline import (
    SimulatedResultTimeline,
    SimulatedWorkerCapacity,
)
from latency_bench.latency.samplers import build_episode_sampler_factory, build_sampler
from latency_bench.scheduler.factory import build_action_scheduler


class ActionLatencyBatch:
    """Schedule independent action replicas without advancing simulator physics.

    Submit at decision boundaries, call ``actions`` before each control step,
    and ``advance`` after it. Actions stay on their original torch device.
    The native trainer owns rewards, terminal observations and episode resets.
    """

    def __init__(self, config, *, num_envs, device, noop_command):
        self.num_envs = num_envs
        self.device = device
        self.clock = EnvClock(
            env_fps=config["env"]["env_fps"], obs_fps=config["env"]["obs_fps"]
        )
        self.noop_command = torch.as_tensor(noop_command, device=device, dtype=torch.float32)
        self.frames = [0] * num_envs
        self.episodes = [-1] * num_envs
        self.submitted = [0] * num_envs
        self.dropped = [0] * num_envs
        self.last_dropped = [False] * num_envs
        self.last_submission = [None] * num_envs
        self.last_application = [None] * num_envs
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
        self.capacities = [
            SimulatedWorkerCapacity(config["executor"]["simulated_worker_capacity"], {})
            for _ in range(num_envs)
        ]
        self.timelines = [
            SimulatedResultTimeline(ordering_policy=OrderingPolicy(config["scheduler"]["ordering_policy"]))
            for _ in range(num_envs)
        ]
        self.schedulers = [
            build_action_scheduler(
                config["scheduler"], clock=self.clock,
                noop_action=Action(self.noop_command, name="noop", is_noop=True),
            )
            for _ in range(num_envs)
        ]

    def _ids(self, env_ids):
        return range(self.num_envs) if env_ids is None else env_ids

    def reset(self, env_ids=None):
        next_episode_id = max(self.episodes) + 1
        for env_id in self._ids(env_ids):
            self.frames[env_id] = 0
            # Like batched evaluation, IID streams follow globally assigned
            # episode IDs; temporal streams remain attached to their replica.
            self.episodes[env_id] = next_episode_id
            next_episode_id += 1
            self.submitted[env_id] = 0
            self.dropped[env_id] = 0
            self.last_dropped[env_id] = False
            self.last_submission[env_id] = None
            self.last_application[env_id] = None
            self.capacities[env_id].reset()
            self.timelines[env_id].reset()
            self.schedulers[env_id].reset()
            if self.sampler_factory is not None:
                self.samplers[env_id] = self.sampler_factory(self.episodes[env_id])

    def submit(self, commands, env_ids=None):
        """Admit actions for a full batch; return a device boolean admission mask."""
        admitted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for env_id in self._ids(env_ids):
            self.last_dropped[env_id] = False
            self.last_submission[env_id] = None
            frame = self.frames[env_id]
            if frame % self.clock.obs_stride_raw_frames:
                continue
            capacity = self.capacities[env_id]
            capacity.release_ready(frame)
            if not capacity.can_submit():
                self.dropped[env_id] += 1
                self.last_dropped[env_id] = True
                continue
            worker = capacity.assign_worker()
            obs_id = self.submitted[env_id]
            sample = self.samplers[env_id].sample({
                "episode_id": self.episodes[env_id], "worker_slot": worker,
                "slot_id": env_id, "obs_id": obs_id, "raw_frame": frame,
                "env_step": frame, "sim_time_ms": self.clock.step_to_time_ms(frame),
                "in_flight_count": capacity.in_flight_count,
                "idle_worker_count": capacity.idle_worker_count,
                "recent_drop_count": self.dropped[env_id],
            })
            value = commands[env_id].detach().clone()
            chunk = value if value.ndim == 2 else None
            command = value[0] if chunk is not None else value
            ready = frame + sample.action_ready_raw_frames
            event = ActionEvent(
                action_id=obs_id, action=Action(command), obs_env_step=frame,
                ready_time_ms=self.clock.step_to_time_ms(ready),
                ready_step_float=ready, ready_raw_frame=ready,
                latency_raw_frames=sample.action_ready_raw_frames,
                latency=LatencyRecord(obs_id=obs_id, latency_ms=sample.latency_ms),
                action_chunk=chunk,
                metadata={"worker_id": worker, "frame_ms": self.clock.frame_ms},
            )
            capacity.submit(worker, frame + sample.worker_service_raw_frames)
            self.timelines[env_id].submit(obs_id=obs_id, ready_raw_frame=ready, event=event)
            self.last_submission[env_id] = {
                "episode_id": self.episodes[env_id], "obs_id": obs_id,
                "issued_raw_frame": frame, "ready_raw_frame": ready,
                "latency_ms": sample.latency_ms, "worker_id": worker,
            }
            self.submitted[env_id] += 1
            admitted[env_id] = True
        return admitted

    def actions(self, env_ids=None):
        """Return actions and record their sources in ``last_application``."""
        result = self.noop_command.expand(self.num_envs, *self.noop_command.shape).clone()
        for env_id in self._ids(env_ids):
            frame = self.frames[env_id]
            released, _ = self.timelines[env_id].release_arrived(frame)
            for event in released:
                self.schedulers[env_id].enqueue(event)
            action = self.schedulers[env_id].update(frame, self.clock.step_to_time_ms(frame))
            result[env_id] = action.value
            event = self.schedulers[env_id].latest_applied_event
            # -1 denotes a real noop, not a missing command record. A held
            # exhausted chunk still belongs to its final element.
            self.last_application[env_id] = {
                "source_raw_frame": -1 if action.is_noop else event.obs_env_step,
                "source_obs_id": -1 if action.is_noop else event.action_id,
                "chunk_index": -1 if action.is_noop else (
                    0 if event.action_chunk is None
                    else min(frame - event.obs_env_step, len(event.action_chunk) - 1)
                ),
            }
        return result

    def advance(self, env_ids=None):
        for env_id in self._ids(env_ids):
            self.frames[env_id] += 1

    def state_dict(self):
        """Save random streams; continuation starts fresh simulator episodes."""
        return {"samplers": deepcopy(self.samplers), "episodes": list(self.episodes)}

    def load_state_dict(self, state):
        self.samplers = deepcopy(state["samplers"])
        self.episodes = list(state["episodes"])
        for env_id in range(self.num_envs):
            self.frames[env_id] = 0
            self.submitted[env_id] = 0
            self.dropped[env_id] = 0
            self.last_dropped[env_id] = False
            self.last_submission[env_id] = None
            self.last_application[env_id] = None
            self.capacities[env_id].reset()
            self.timelines[env_id].reset()
            self.schedulers[env_id].reset()


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
