"""Parent handle for the environment process shared by warmup and evaluation."""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing import resource_tracker

from latency_bench.executors.env_step_backend import resolve_start_method
from latency_bench.executors.realtime.env_worker import _env_loop_worker
from latency_bench.executors.thread_env import set_thread_env_defaults


class RealtimeEnvProcess:
    """Own one environment worker, its shared observations, and run lifetime."""

    def __init__(self, config, *, video_output_dir=None, video_fps=None):
        set_thread_env_defaults()
        # Forked env and spawned inference workers must share the same tracker.
        resource_tracker.ensure_running()
        context = mp.get_context(resolve_start_method())
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_env_loop_worker,
            args=(child, config, video_output_dir, video_fps),
            name="latency-bench-realtime-env",
        )
        self.closed = False
        self.process.start()
        child.close()
        try:
            metadata = self.receive()
            config["env"].update(metadata["env_config"])
            self.env_fps = metadata["env_fps"]
            self.noop_action = metadata["noop_action"]
            self.action_space = metadata["action_space"]
        except BaseException:
            self.close()
            raise

    def receive(self):
        tag, payload = self.connection.recv()
        if tag == "error":
            raise RuntimeError(payload["traceback"])
        return payload

    def reset(self, seed=None):
        self.connection.send(("reset", {"seed": seed}))
        self.receive()

    def warmup_observation(self):
        self.connection.send(("observe", {}))
        return self.receive()["task"]

    def release_observation(self, obs_id):
        self.connection.send(("release_observation", {"obs_id": obs_id}))

    def step(self, action):
        self.connection.send(("step", {"action": action}))
        return self.receive()["result"]

    def start_episode(self, *, episode_id, seed, max_env_steps):
        self.connection.send(("episode", {
            "episode_id": episode_id, "seed": seed, "max_env_steps": max_env_steps,
        }))
        return self.receive()["episode_start_s"]

    def finish_episode(self):
        self.connection.send(("finish_episode", {}))
        self.receive()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.connection.send(("close", {}))
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        self.connection.close()
