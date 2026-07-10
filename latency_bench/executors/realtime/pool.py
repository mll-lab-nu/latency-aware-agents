"""Parent-side concurrency machinery: the inference process pool."""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

from latency_bench.executors.env_step_backend import _set_thread_env_defaults
from latency_bench.executors.realtime.messages import (
    RealtimeObservationTask,
    RealtimePolicyCompletion,
)
from latency_bench.executors.realtime.workers import (
    _inference_worker,
    _profile_now_ms,
)

REALTIME_INFERENCE_START_METHOD = "spawn"


class ProcessInferencePool:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        inference_devices: list[str],
    ):
        _set_thread_env_defaults()
        self.config = config
        self.inference_devices = list(inference_devices)
        self.start_method = REALTIME_INFERENCE_START_METHOD
        self._ctx = mp.get_context(self.start_method)
        self._connections: list[Any] = []
        self._processes: list[mp.Process] = []
        self._busy_worker_tasks: dict[int, RealtimeObservationTask] = {}
        self.closed = False
        self._start_workers()

    @property
    def in_flight_count(self) -> int:
        return len(self._busy_worker_tasks)

    @property
    def idle_worker_count(self) -> int:
        return len(self.inference_devices) - len(self._busy_worker_tasks)

    @property
    def worker_count(self) -> int:
        return len(self.inference_devices)

    def drain_until_idle(self) -> dict[str, float | int]:
        self._raise_if_closed()
        busy_before = self.in_flight_count
        wait_start_s = time.perf_counter()
        drained_completion_count = 0
        while self.in_flight_count:
            completions = self._drain_all_ready()
            drained_completion_count += len(completions)
            if self.in_flight_count:
                time.sleep(0.001)
        # Workers run with automatic GC disabled; every worker is idle right
        # now, so this is the one safe moment to reclaim their cycle garbage.
        # Pipe ordering guarantees the collect finishes before any later
        # predict is processed.
        for connection in self._connections:
            connection.send(("collect", {}))
        return {
            "pre_episode_in_flight_count": int(busy_before),
            "pre_episode_drained_completion_count": int(drained_completion_count),
            "pre_episode_drain_wait_ms": float((time.perf_counter() - wait_start_s) * 1000.0),
            "pre_episode_in_flight_after_drain": int(self.in_flight_count),
        }

    def submit(self, task: RealtimeObservationTask) -> bool:
        self._raise_if_closed()
        for worker_id in range(len(self.inference_devices)):
            if worker_id not in self._busy_worker_tasks:
                if task.profile is not None:
                    task.profile.parent_to_worker_send_start_ms = _profile_now_ms()
                self._connections[worker_id].send(("predict", {"task": task}))
                if task.profile is not None:
                    task.profile.parent_to_worker_send_call_ms = (
                        _profile_now_ms() - task.profile.parent_to_worker_send_start_ms
                    )
                self._busy_worker_tasks[worker_id] = task
                return True
        return False

    def drain_completed(self, now_ms: float) -> list[RealtimePolicyCompletion]:
        del now_ms
        self._raise_if_closed()
        return self._drain_all_ready()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", {}))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for process in self._processes:
            process.join(timeout=0.2)
        terminated_processes = []
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                terminated_processes.append(process)
        for process in terminated_processes:
            process.join(timeout=1.0)
        for process in self._processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass

    def _start_workers(self) -> None:
        try:
            for worker_id, device in enumerate(self.inference_devices):
                parent_conn, child_conn = self._ctx.Pipe()
                process = self._ctx.Process(
                    target=_inference_worker,
                    args=(child_conn, self.config, worker_id, device, self.start_method),
                    name=f"latency-bench-realtime-inference-{worker_id}",
                )
                process.start()
                child_conn.close()
                self._connections.append(parent_conn)
                self._processes.append(process)

            for worker_id, connection in enumerate(self._connections):
                tag, payload = connection.recv()
                if tag == "error":
                    raise RuntimeError(payload.get("traceback", "<missing traceback>"))
                if tag != "ready":
                    raise RuntimeError(f"inference worker {worker_id} returned {tag!r} during startup")
        except Exception:
            self.close()
            raise

    def _drain_all_ready(self) -> list[RealtimePolicyCompletion]:
        completed: list[RealtimePolicyCompletion] = []
        for worker_id, connection in enumerate(self._connections):
            while connection.poll():
                tag, payload = connection.recv()
                if tag == "result":
                    task = self._busy_worker_tasks.pop(worker_id)
                    result = payload["result"]
                    profile = task.profile
                    if profile is not None and result.profile is not None:
                        profile.merge_measured(result.profile)
                        profile.worker_to_parent_ipc_ms = (
                            _profile_now_ms() - profile.worker_result_send_start_ms
                        )
                    completed.append(
                        RealtimePolicyCompletion(
                            task=task,
                            policy_output=result.policy_output,
                            completion_time_ms=result.completion_time_ms,
                            latency_record=result.latency_record,
                            worker_id=worker_id,
                            device=self.inference_devices[worker_id],
                            profile=profile,
                        )
                    )
                elif tag == "error":
                    self._busy_worker_tasks.pop(worker_id, None)
                    raise RuntimeError(payload.get("traceback", "<missing traceback>"))
        return completed

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("inference pool is closed")

