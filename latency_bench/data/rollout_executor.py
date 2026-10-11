"""Adapt existing raw-frame executors to canonical decision sample collection."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from latency_bench.executors.base import BatchedExecutor, CollectionObserver
from latency_bench.executors.factory import build_executor
from latency_bench.eval.config import _evaluation_seed
from latency_bench.utils.array import to_numpy
from latency_bench.utils.seeding import seed_everything



class ExecutorCollectionObserver(CollectionObserver):
    """Pair source snapshots with asynchronous teacher outputs and interval rewards."""

    def __init__(self, session, student):
        self.session = session
        self.student = student
        self.capture_rgb = session.capture and student["image_key"] is None
        self.episodes = {}

    @property
    def stop_requested(self):
        return self.session.target_reached

    def start_episode(self, episode_id, seed, slot_id):
        self.episodes[episode_id] = {"requests": {}, "intervals": {}, "steps": 0,
                                    "reward": 0.0, "decisions": 0, "info": {}, "terminated": False, "truncated": False}

    def observation(self, episode_id, *, obs_id, observation, image=None, latency_ms=None, latency_raw_frames=None):
        if not self.session.capture:
            return
        # Explicit frames already own their storage. Mapping inputs can reuse buffers.
        if image is None:
            image = np.asarray(to_numpy(observation.data[self.student["image_key"]])).copy()
        state_key = self.student["state_key"]
        state_source = observation.data if self.student["state_source"] == "data" else observation.metadata
        state = None if state_key is None else np.asarray(to_numpy(state_source[state_key])).copy()
        request = {
            "decision_step": observation.env_step // self.session.clock_stride,
            "image": image, "state": state, "reward": 0.0,
            "info": {"latency_ms": observation.metadata["latency_ms"] if latency_ms is None else latency_ms,
                     "latency_raw_frames": observation.metadata["latency_raw_frames"] if latency_raw_frames is None else latency_raw_frames},
            "complete": False,
        }
        episode = self.episodes[episode_id]
        episode["requests"][obs_id] = request
        episode["intervals"][request["decision_step"]] = obs_id

    def output(self, episode_id, *, obs_id, output):
        self.episodes[episode_id]["decisions"] += 1
        if not self.session.capture:
            return
        if output.action_chunk is not None:
            raise ValueError("Generic canonical collection supports single-action labels; use the task's chunk recipe")
        request = self.episodes[episode_id]["requests"][obs_id]
        values = self.session.task["action_values"]
        request["issued_action"] = (values.index(output.action.value) if values is not None
                                    else deepcopy(output.action.value))
        self._emit(episode_id, obs_id)

    def step(self, record, *, terminated, truncated):
        episode = self.episodes[record.episode_id]
        episode["steps"] = record.env_step + 1
        episode["reward"] += float(record.reward)
        episode.update(terminated=terminated, truncated=truncated, info=record.info)
        interval = record.env_step // self.session.clock_stride
        obs_id = episode["intervals"].get(interval)
        if obs_id is not None:
            request = episode["requests"][obs_id]
            request["reward"] += float(record.reward)
            request["complete"] = terminated or truncated or (record.env_step + 1) % self.session.clock_stride == 0
            self._emit(record.episode_id, obs_id)

    def _emit(self, episode_id, obs_id):
        episode = self.episodes[episode_id]
        request = episode["requests"][obs_id]
        if request["complete"] and "issued_action" in request:
            self.session.decision(episode_id, **{key: value for key, value in request.items() if key != "complete"})
            del episode["intervals"][request["decision_step"]]
            del episode["requests"][obs_id]

    def finish_episode(self, metrics):
        episode_id = metrics.episode_id
        episode = self.episodes[episode_id]
        for obs_id in list(episode["requests"]):
            request = episode["requests"][obs_id]
            request["complete"] = True
            self._emit(episode_id, obs_id)
        summary = {
            "episode_length": episode["decisions"],
            "episode_raw_frames": episode["steps"],
            self.session.task["action_spec"].episode_return_field: episode["reward"],
            "termination_reason": "terminated" if episode["terminated"] else "truncated" if episode["truncated"] else "horizon",
            "final_info": episode["info"],
        }
        self.session.complete_episode(episode_id, summary)
        del self.episodes[episode_id]


class ExecutorCollectionExecution:
    """Run existing executors with shared per-slot episode allocation and sample capture."""

    def __init__(self, config, *, num_envs):
        self.config = deepcopy(config)
        self.config["executor"]["parallel_envs"] = num_envs
        seed_everything(_evaluation_seed(self.config))
        env = self.config["env"]
        self.metadata = {"env_fps": env["env_fps"], "obs_fps": env["obs_fps"],
                         "frame_stack": env["frame_stack"] if "frame_stack" in env else 1}
        self.executor = build_executor(self.config)

    def run(self, session):
        observer = ExecutorCollectionObserver(session, self.config["collection"]["student_observation"])
        self.executor.collection_observer = observer
        self.executor.run_warmup()
        if isinstance(self.executor, BatchedExecutor):
            self.executor.run_episodes(episode_ids=[], seeds=[],
                max_env_steps=session.options["max_steps_per_episode"],
                next_episode=session.next_episode, on_episode_complete=observer.finish_episode)
        else:
            while (spec := session.next_episode()) is not None:
                metrics = self.executor.run_episode(episode_id=spec["episode_idx"], seed=spec["seed"],
                                                     max_env_steps=session.options["max_steps_per_episode"])
                observer.finish_episode(metrics)

    def close(self):
        self.executor.close()
