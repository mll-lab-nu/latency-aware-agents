"""Parent-side concurrency machinery: the inference process pool."""
from __future__ import annotations

import json
import multiprocessing as mp
import multiprocessing.spawn as mp_spawn
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from typing import Any

from latency_bench.core.types import Observation, PolicyOutput
from latency_bench.executors.realtime.inference_worker import _inference_worker
from latency_bench.executors.realtime.messages import (
    RealtimeObservationTask,
    RealtimePolicyCompletion,
)
from latency_bench.executors.realtime.timing import monotonic_ms
from latency_bench.executors.thread_env import set_thread_env_defaults
from latency_bench.utils.paths import REPO_ROOT

REALTIME_INFERENCE_START_METHOD = "spawn"
_PROCESS_START_LOCK = threading.Lock()


class ProcessInferencePool:
    """Own inference subprocesses and preserve logical policy-slot state."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        inference_devices: list[str],
    ):
        set_thread_env_defaults()
        self.config = config
        self.inference_devices = list(inference_devices)
        self.start_method = REALTIME_INFERENCE_START_METHOD
        self._ctx = mp.get_context(self.start_method)
        self._connections: list[Any] = []
        self._processes: list[mp.Process] = []
        self._busy_worker_tasks: dict[int, tuple[RealtimeObservationTask, float]] = {}
        self.closed = False
        self.worker_python_executable = None
        self.worker_sys_path = None
        if "policy" in config and "worker_python_executable" in config["policy"]:
            self.worker_python_executable = config["policy"]["worker_python_executable"]
            self.worker_sys_path = json.loads(
                subprocess.check_output(
                    [
                        self.worker_python_executable,
                        "-I",
                        "-c",
                        "import json, sys; print(json.dumps(sys.path))",
                    ],
                    text=True,
                )
            )
            # A wheel's parent also contains this interpreter's binary dependencies.
            if (REPO_ROOT / "pyproject.toml").is_file():
                self.worker_sys_path.insert(0, str(REPO_ROOT))
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
                worker_service_start_time_ms = monotonic_ms()
                self._connections[worker_id].send(("predict", {"task": task}))
                self._busy_worker_tasks[worker_id] = (
                    task,
                    worker_service_start_time_ms,
                )
                return True
        return False

    def predict_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        """Run synchronous batches with stable slot-to-worker ownership."""
        self._raise_if_closed()
        indexed_by_worker: list[list[tuple[int, Observation]]] = [
            [] for _worker_id in self.inference_devices
        ]
        for index, observation in enumerate(observations):
            worker_id = observation.metadata["slot_id"] % self.worker_count
            indexed_by_worker[worker_id].append((index, observation))

        batch_size = self.config["executor"]["inference_batch_size"]
        chunks_by_worker = [
            [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
            for items in indexed_by_worker
        ]
        outputs: list[PolicyOutput | None] = [None] * len(observations)
        while any(chunks_by_worker):
            active: list[tuple[int, list[tuple[int, Observation]]]] = []
            for worker_id, chunks in enumerate(chunks_by_worker):
                if not chunks:
                    continue
                chunk = chunks.pop(0)
                self._connections[worker_id].send(
                    ("predict_batch", {"observations": [item[1] for item in chunk]})
                )
                active.append((worker_id, chunk))
            for worker_id, chunk in active:
                tag, payload = self._connections[worker_id].recv()
                if tag == "error":
                    raise RuntimeError(payload["traceback"])
                if tag != "batch_result":
                    raise RuntimeError(f"inference worker returned {tag!r} for predict_batch")
                for (index, _observation), output in zip(chunk, payload["policy_outputs"], strict=True):
                    outputs[index] = output
        return outputs  # type: ignore[return-value]

    def reset_state(self, slot_id: int) -> None:
        """Reset one logical slot on every worker process."""
        self._raise_if_closed()
        for connection in self._connections:
            connection.send(("reset_state", {"slot_id": slot_id}))
        for connection in self._connections:
            tag, payload = connection.recv()
            if tag == "error":
                raise RuntimeError(payload["traceback"])
            if tag != "state_reset":
                raise RuntimeError(f"inference worker returned {tag!r} for reset_state")

    def drain_completed(self) -> list[RealtimePolicyCompletion]:
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
            with _PROCESS_START_LOCK:
                previous_executable = mp_spawn.get_executable()
                previous_sys_path = sys.path.copy()
                try:
                    if self.worker_python_executable is not None:
                        sys.path[:] = self.worker_sys_path
                        mp.set_executable(self.worker_python_executable)
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
                finally:
                    sys.path[:] = previous_sys_path
                    mp.set_executable(previous_executable)

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
                    result_received_time_ms = monotonic_ms()
                    task, worker_service_start_time_ms = self._busy_worker_tasks.pop(
                        worker_id
                    )
                    result = payload["result"]
                    completed.append(
                        RealtimePolicyCompletion(
                            task=task,
                            policy_output=result.policy_output,
                            completion_time_ms=result.completion_time_ms,
                            latency_record=result.latency_record,
                            observation_to_action_latency_ms=(
                                result_received_time_ms - task.observation_start_time_ms
                            ),
                            worker_service_time_ms=(
                                result_received_time_ms - worker_service_start_time_ms
                            ),
                            worker_id=worker_id,
                            device=self.inference_devices[worker_id],
                        )
                    )
                elif tag == "error":
                    self._busy_worker_tasks.pop(worker_id, None)
                    raise RuntimeError(payload.get("traceback", "<missing traceback>"))
        return completed

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("inference pool is closed")
