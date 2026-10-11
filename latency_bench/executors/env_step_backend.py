from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter, collection_frame
from latency_bench.envs.factory import build_env_adapter
from latency_bench.executors.thread_env import set_thread_env_defaults


START_METHOD_ENV = "LATENCY_BENCH_ENV_BACKEND_START_METHOD"
WORKER_TIMEOUT_ENV = "LATENCY_BENCH_ENV_WORKER_TIMEOUT_SEC"


@dataclass(frozen=True)
class EnvStepResponse:
    """Environment result and metadata.

    ``*_wall_sec`` fields are ``perf_counter`` duration seconds.
    """

    result: StepResult
    worker_pid: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EnvStepBackend(Protocol):
    slot_handles: Sequence[EnvAdapter]
    last_step_metadata_by_slot: dict[int, dict[str, Any]]
    collection_frames: dict[int, Any]
    closed: bool

    @property
    def num_slots(self) -> int:
        ...

    @property
    def env_fps(self) -> float:
        ...

    def worker_pids(self) -> list[int]:
        ...

    def reset_slot(self, slot_id: int, *, episode_id: int, seed: int | None) -> Observation:
        ...

    def observe_slots(self, slot_ids: Sequence[int], *, capture_rgb: bool = False) -> dict[int, Observation]:
        ...

    def step_slots(self, actions_by_slot: Mapping[int, Action]) -> dict[int, EnvStepResponse]:
        ...

    def close(self) -> None:
        ...


class EnvStepBackendEnvAdapter(EnvAdapter):
    """Expose slot zero of a task backend to the realtime executor."""

    def __init__(self, backend: EnvStepBackend):
        self.backend = backend
        handle = backend.slot_handles[0]
        self.gym_env = handle.gym_env
        self.env_fps = backend.env_fps
        self.noop_action = handle.noop_action
        self.action_space = handle.action_space
        self._episode_id = 0

    def reset(self, seed: int | None = None) -> Observation:
        observation = self.backend.reset_slot(
            0,
            episode_id=self._episode_id,
            seed=seed,
        )
        self._episode_id += 1
        return observation

    def observe(self) -> Observation:
        return self.backend.observe_slots([0])[0]

    def step(self, action: Action) -> StepResult:
        return self.backend.step_slots({0: action})[0].result

    def close(self) -> None:
        self.backend.close()


class RemoteEnvSlotHandle(EnvAdapter):
    """Main-process handle for an env owned by a worker process."""

    def __init__(
        self,
        *,
        slot_id: int,
        env_fps: float,
        noop_action: Action,
        action_space: Any = None,
    ):
        self.slot_id = int(slot_id)
        self.env_fps = float(env_fps)
        self.noop_action = noop_action
        self.action_space = action_space
        self.gym_env = self
        self._last_observation: Observation | None = None
        self._last_info: dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> Observation:
        raise RuntimeError("remote env slots must be reset through EnvStepBackend")

    def step(self, action: Action) -> StepResult:
        raise RuntimeError("remote env slots must be stepped through EnvStepBackend")

    def observe(self) -> Observation:
        if self._last_observation is None:
            raise RuntimeError("remote env slots must be observed through EnvStepBackend")
        return self._last_observation

    def update_observation(self, observation: Observation | None, info: dict[str, Any] | None = None) -> None:
        if observation is not None:
            self._last_observation = observation
        if info is not None:
            self._last_info = dict(info)

    def render_frame(self) -> Any:
        if self._last_observation is None:
            return None
        return self._last_observation.data

    def get_state_info(self) -> dict[str, Any]:
        if self._last_observation is None:
            return dict(self._last_info)
        return {
            **self._last_info,
            **self._last_observation.metadata,
            "env_step": self._last_observation.env_step,
            "sim_time_ms": self._last_observation.sim_time_ms,
        }


class SerialEnvStepBackend:
    def __init__(self, envs: Sequence[EnvAdapter]):
        self.slot_handles = list(envs)
        self.last_step_metadata_by_slot: dict[int, dict[str, Any]] = {}
        self.collection_frames: dict[int, Any] = {}
        self.closed = False

    @property
    def num_slots(self) -> int:
        return len(self.slot_handles)

    @property
    def env_fps(self) -> float:
        return float(self.slot_handles[0].env_fps)

    def worker_pids(self) -> list[int]:
        return []

    def reset_slot(self, slot_id: int, *, episode_id: int, seed: int | None) -> Observation:
        self._raise_if_closed()
        env = self.slot_handles[int(slot_id)]
        observation = env.reset(seed=seed)
        return observation

    def observe_slots(self, slot_ids: Sequence[int], *, capture_rgb: bool = False) -> dict[int, Observation]:
        self._raise_if_closed()
        observations = {}
        for slot_id in slot_ids:
            env = self.slot_handles[int(slot_id)]
            obs = env.observe()
            if capture_rgb:
                self.collection_frames[int(slot_id)] = collection_frame(env, obs)
            observations[int(slot_id)] = obs
        return observations

    def step_slots(self, actions_by_slot: Mapping[int, Action]) -> dict[int, EnvStepResponse]:
        self._raise_if_closed()
        responses: dict[int, EnvStepResponse] = {}
        for slot_id, action in actions_by_slot.items():
            slot_id = int(slot_id)
            env = self.slot_handles[slot_id]
            start = time.perf_counter()
            result = env.step(action)
            metadata = {
                "backend": "serial",
                "slot_id": slot_id,
                "worker_pid": os.getpid(),
                "worker_step_wall_sec": time.perf_counter() - start,
            }
            self.last_step_metadata_by_slot[slot_id] = metadata
            responses[slot_id] = EnvStepResponse(
                result=result,
                worker_pid=os.getpid(),
                metadata=metadata,
            )
        return responses

    def close(self) -> None:
        self.collection_frames.clear()
        if self.closed:
            return
        self.closed = True
        for env in self.slot_handles:
            env.close()

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("env step backend is closed")


class SubprocEnvStepBackend:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        noop_action: Action,
        num_slots: int,
        start_method: str | None = None,
        timeout_sec: float | None = None,
    ):
        set_thread_env_defaults()
        self.config = config
        self.noop_action = noop_action
        self.timeout_sec = _worker_timeout_sec() if timeout_sec is None else float(timeout_sec)
        self.start_method = resolve_start_method(start_method)
        self._ctx = mp.get_context(self.start_method)
        self._connections: list[Any] = []
        self._processes: list[mp.Process] = []
        self.slot_handles: list[RemoteEnvSlotHandle] = []
        self.last_step_metadata_by_slot: dict[int, dict[str, Any]] = {}
        self.collection_frames: dict[int, Any] = {}
        self.closed = False
        self._start_workers(int(num_slots))

    @property
    def num_slots(self) -> int:
        return len(self.slot_handles)

    @property
    def env_fps(self) -> float:
        return float(self.slot_handles[0].env_fps)

    def worker_pids(self) -> list[int]:
        return [int(process.pid) for process in self._processes if process.pid is not None]

    def reset_slot(self, slot_id: int, *, episode_id: int, seed: int | None) -> Observation:
        self._raise_if_closed()
        slot_id = int(slot_id)
        self._send(slot_id, "reset", {"episode_id": int(episode_id), "seed": seed})
        payload = self._recv(slot_id, "reset_result")
        observation = payload["observation"]
        metadata = dict(payload.get("metadata", {}))
        handle = self.slot_handles[slot_id]
        handle.update_observation(observation, metadata)
        return observation

    def observe_slots(self, slot_ids: Sequence[int], *, capture_rgb: bool = False) -> dict[int, Observation]:
        self._raise_if_closed()
        slot_ids = [int(slot_id) for slot_id in slot_ids]
        for slot_id in slot_ids:
            self._send(slot_id, "observe", {"capture_rgb": True} if capture_rgb else {})
        observations: dict[int, Observation] = {}
        for slot_id in slot_ids:
            payload = self._recv(slot_id, "observe_result")
            observation = payload["observation"]
            if capture_rgb:
                self.collection_frames[slot_id] = payload["collection_frame"]
            self.slot_handles[slot_id].update_observation(observation, None)
            observations[slot_id] = observation
        return observations

    def step_slots(self, actions_by_slot: Mapping[int, Action]) -> dict[int, EnvStepResponse]:
        self._raise_if_closed()
        items = [(int(slot_id), action) for slot_id, action in actions_by_slot.items()]
        for slot_id, action in items:
            self._send(slot_id, "step", {"action": _step_payload_from_action(action)})

        responses: dict[int, EnvStepResponse] = {}
        for slot_id, _action in items:
            payload = self._recv(slot_id, "step_result")
            result = payload["result"]
            metadata = dict(payload.get("metadata", {}))
            if result.observation is not None:
                self.slot_handles[slot_id].update_observation(result.observation, result.info)
            else:
                self.slot_handles[slot_id].update_observation(None, result.info)
            self.last_step_metadata_by_slot[slot_id] = metadata
            responses[slot_id] = EnvStepResponse(
                result=result,
                worker_pid=int(payload.get("worker_pid")),
                metadata=metadata,
            )
        return responses

    def close(self) -> None:
        self.collection_frames.clear()
        self._shutdown_workers(terminate=False)

    def _start_workers(self, num_slots: int) -> None:
        try:
            for slot_id in range(num_slots):
                parent_conn, child_conn = self._ctx.Pipe()
                process = self._ctx.Process(
                    target=_env_worker,
                    args=(child_conn, self.config, self.noop_action, slot_id),
                    name=f"latency-bench-env-slot-{slot_id}",
                )
                process.start()
                child_conn.close()
                self._connections.append(parent_conn)
                self._processes.append(process)

            for slot_id in range(num_slots):
                payload = self._recv(slot_id, "ready")
                self.slot_handles.append(
                    RemoteEnvSlotHandle(
                        slot_id=slot_id,
                        env_fps=float(payload["env_fps"]),
                        noop_action=payload["noop_action"],
                        action_space=payload.get("action_space"),
                    )
                )
        except Exception:
            self._shutdown_workers(terminate=True)
            raise

    def _send(self, slot_id: int, command: str, payload: dict[str, Any] | None = None) -> None:
        if self.closed:
            raise RuntimeError("env step backend is closed")
        try:
            self._connections[slot_id].send((command, payload or {}))
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._fail(f"subprocess env worker {slot_id} is unavailable while sending {command}: {exc}")

    def _recv(self, slot_id: int, expected_tag: str) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("env step backend is closed")
        connection = self._connections[slot_id]
        try:
            if not connection.poll(self.timeout_sec):
                self._fail(
                    f"subprocess env worker {slot_id} timed out waiting for {expected_tag} "
                    f"after {self.timeout_sec:.1f}s"
                )
            tag, payload = connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._fail(f"subprocess env worker {slot_id} failed while receiving {expected_tag}: {exc}")

        if tag == "error":
            trace = payload.get("traceback", "<missing traceback>")
            self._fail(f"subprocess env worker {slot_id} failed:\n{trace}")
        if tag != expected_tag:
            self._fail(
                f"subprocess env worker {slot_id} returned unexpected message {tag!r}; "
                f"expected {expected_tag!r}"
            )
        return payload

    def _fail(self, message: str) -> None:
        self._shutdown_workers(terminate=True)
        raise RuntimeError(message)

    def _shutdown_workers(self, *, terminate: bool) -> None:
        if self.closed:
            return
        self.closed = True
        if not terminate:
            for connection, process in zip(self._connections, self._processes):
                if process.is_alive():
                    try:
                        connection.send(("close", {}))
                    except (BrokenPipeError, EOFError, OSError):
                        terminate = True

            deadline = time.monotonic() + self.timeout_sec
            for connection, process in zip(self._connections, self._processes):
                if not process.is_alive():
                    continue
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    if remaining > 0.0 and connection.poll(remaining):
                        connection.recv()
                    else:
                        terminate = True
                except (BrokenPipeError, EOFError, OSError):
                    terminate = True

        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass

        for process in self._processes:
            process.join(timeout=0.2)

        if terminate:
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
            for process in self._processes:
                process.join(timeout=1.0)

        for process in self._processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("env step backend is closed")


def _env_worker(connection: Any, config: dict[str, Any], noop_action: Action, slot_id: int) -> None:
    env: EnvAdapter | None = None
    worker_pid = os.getpid()
    try:
        env = build_env_adapter(config, noop_action)
        connection.send((
            "ready",
            {
                "slot_id": int(slot_id),
                "worker_pid": worker_pid,
                "env_fps": env.env_fps,
                "noop_action": env.noop_action,
                "action_space": env_action_space(env),
            },
        ))

        while True:
            command, payload = connection.recv()
            payload = payload or {}
            if command == "reset":
                start = time.perf_counter()
                observation = env.reset(seed=payload.get("seed"))
                metadata = {
                    "backend": "subproc",
                    "slot_id": int(slot_id),
                    "episode_id": int(payload["episode_id"]),
                    "worker_pid": worker_pid,
                    "worker_reset_wall_sec": time.perf_counter() - start,
                }
                connection.send((
                    "reset_result",
                    {
                        "observation": observation,
                        "worker_pid": worker_pid,
                        "metadata": metadata,
                    },
                ))
            elif command == "observe":
                observation = env.observe()
                response = {"observation": observation, "worker_pid": worker_pid}
                if payload.get("capture_rgb"):
                    response["collection_frame"] = collection_frame(env, observation)
                connection.send(("observe_result", response))
            elif command == "step":
                start = time.perf_counter()
                result = env.step(_action_from_step_payload(payload["action"]))
                parent_result = StepResult(
                    observation=None,
                    reward=result.reward,
                    done=result.done,
                    truncated=result.truncated,
                    info=result.info,
                )
                metadata = {
                    "backend": "subproc",
                    "slot_id": int(slot_id),
                    "worker_pid": worker_pid,
                    "worker_step_wall_sec": time.perf_counter() - start,
                }
                connection.send((
                    "step_result",
                    {
                        "result": parent_result,
                        "worker_pid": worker_pid,
                        "metadata": metadata,
                    },
                ))
            elif command == "close":
                connection.send(("close_ack", {"worker_pid": worker_pid}))
                break
            else:
                raise ValueError(f"unknown env worker command: {command}")
    except BaseException:
        try:
            connection.send((
                "error",
                {
                    "slot_id": int(slot_id),
                    "worker_pid": worker_pid,
                    "traceback": traceback.format_exc(),
                },
            ))
        except BaseException:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        try:
            connection.close()
        except BaseException:
            pass

def _worker_timeout_sec() -> float:
    raw_value = os.environ.get(WORKER_TIMEOUT_ENV)
    if raw_value is None:
        return 60.0
    return float(raw_value)


def resolve_start_method(start_method: str | None = None) -> str:
    available = mp.get_all_start_methods()

    requested = start_method or os.environ.get(START_METHOD_ENV)
    if requested is not None:
        requested = str(requested).strip().lower()
        if requested in available:
            return requested
        if start_method is not None or START_METHOD_ENV in os.environ:
            raise ValueError(f"multiprocessing start method {requested!r} is not available; available={available}")

    for fallback in ("fork", "forkserver", "spawn"):
        if fallback in available:
            return fallback
    raise RuntimeError(f"no safe multiprocessing start method is available; available={available}")


def env_action_space(env: EnvAdapter) -> Any:
    for owner in (env, getattr(env, "gym_env", None), getattr(env, "sf_env", None)):
        if owner is None:
            continue
        action_space = getattr(owner, "action_space", None)
        if action_space is not None:
            return action_space
    return None


def _step_payload_from_action(action: Action) -> tuple[Any, str | None, bool, bool]:
    return (action.value, action.name, action.is_noop, action.is_oneshot)


def _action_from_step_payload(payload: tuple[Any, str | None, bool, bool]) -> Action:
    value, name, is_noop, is_oneshot = payload
    return Action(value=value, name=name, is_noop=is_noop, is_oneshot=is_oneshot)
