"""IPC between a DAgger controller and StarVLA training ranks at a boundary."""

from __future__ import annotations

import json
import socket
import time
import traceback
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from latency_bench.core.types import Observation, PolicyOutput


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ResidentInferencePool:
    """Route controller batches to the training ranks that own the live model."""

    def __init__(
        self, *, socket_dir: Path, step: int, worker_count: int, batch_size: int
    ):
        self.worker_count = worker_count
        self.batch_size = batch_size
        self.connections = []
        for rank in range(worker_count):
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(str(socket_dir / f"step_{step}.rank_{rank}.sock"))
            self.connections.append(Connection(client.detach()))
        self.closed = False

    def predict_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        indexed_by_worker: list[list[tuple[int, Observation]]] = [
            [] for _rank in range(self.worker_count)
        ]
        for index, observation in enumerate(observations):
            rank = observation.metadata["slot_id"] % self.worker_count
            indexed_by_worker[rank].append((index, observation))

        chunks_by_worker = [
            [items[start : start + self.batch_size] for start in range(0, len(items), self.batch_size)]
            for items in indexed_by_worker
        ]
        outputs: list[PolicyOutput | None] = [None] * len(observations)
        while any(chunks_by_worker):
            active: list[tuple[int, list[tuple[int, Observation]]]] = []
            for rank, chunks in enumerate(chunks_by_worker):
                if chunks:
                    chunk = chunks.pop(0)
                    self.connections[rank].send(
                        ("predict_batch", {"observations": [item[1] for item in chunk]})
                    )
                    active.append((rank, chunk))
            for rank, chunk in active:
                tag, payload = self.connections[rank].recv()
                if tag == "error":
                    raise RuntimeError(payload["traceback"])
                if tag != "batch_result":
                    raise RuntimeError(f"resident inference rank returned {tag!r}")
                for (index, _observation), output in zip(
                    chunk, payload["policy_outputs"], strict=True
                ):
                    outputs[index] = output
        return outputs  # type: ignore[return-value]

    def reset_state(self, slot_id: int | None = None) -> None:
        """Clear a slot on its owning rank, or all rank state at a boundary."""
        ranks = range(self.worker_count) if slot_id is None else [slot_id % self.worker_count]
        for rank in ranks:
            self.connections[rank].send(("reset_state", {"slot_id": slot_id}))
        for rank in ranks:
            tag, payload = self.connections[rank].recv()
            if tag == "error":
                raise RuntimeError(payload["traceback"])
            if tag != "reset_done":
                raise RuntimeError(f"resident inference rank returned {tag!r} for reset_state")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send(("close", {}))
                connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()


def serve_resident_inference(
    *,
    policy: Any,
    control_dir: Path,
    socket_dir: Path,
    step: int,
    rank: int,
    world_size: int,
    stop_requested: Callable[[], bool],
) -> dict[str, Any] | None:
    """Serve one saved boundary until the controller publishes its continue message."""
    # Only the training-side service needs torch; controller IPC remains lightweight.
    import torch

    # Release unused training allocations before the simulator shares this GPU.
    torch.cuda.empty_cache()
    control_dir.mkdir(parents=True, exist_ok=True)
    control_dir.chmod(0o700)
    socket_dir.mkdir(parents=True, exist_ok=True)
    socket_path = socket_dir / f"step_{step}.rank_{rank}.sock"
    rank_ready_path = control_dir / f"step_{step}.rank_{rank}.ready"
    ready_path = control_dir / f"step_{step}.ready"
    continue_path = control_dir / f"step_{step}.continue.json"
    socket_path.unlink(missing_ok=True)
    rank_ready_path.unlink(missing_ok=True)

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen()
    listener.settimeout(0.25)
    _write_json_atomic(rank_ready_path, {"step": step, "rank": rank})
    if rank == 0:
        ready_path.unlink(missing_ok=True)
        rank_ready_paths = [
            control_dir / f"step_{step}.rank_{worker_rank}.ready"
            for worker_rank in range(world_size)
        ]
        while not all(path.is_file() for path in rank_ready_paths):
            if stop_requested():
                listener.close()
                socket_path.unlink(missing_ok=True)
                return None
            time.sleep(0.1)
        _write_json_atomic(ready_path, {"step": step, "workers": world_size})

    connection: Connection | None = None
    try:
        while not continue_path.is_file():
            if stop_requested():
                return None
            if connection is None:
                try:
                    accepted, _address = listener.accept()
                    connection = Connection(accepted.detach())
                except TimeoutError:
                    continue
            if not connection.poll(0.25):
                continue
            try:
                tag, payload = connection.recv()
                if tag == "close":
                    connection.send(("closed", {}))
                    connection.close()
                    connection = None
                elif tag == "predict_batch":
                    outputs = policy.predict_batch(payload["observations"])
                    connection.send(("batch_result", {"policy_outputs": outputs}))
                elif tag == "reset_state":
                    policy.reset_state(payload["slot_id"])
                    connection.send(("reset_done", {}))
                else:
                    connection.send(("error", {"traceback": f"unknown resident request {tag!r}"}))
            except Exception:
                if connection is not None:
                    try:
                        connection.send(("error", {"traceback": traceback.format_exc()}))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                    connection.close()
                    connection = None
        return json.loads(continue_path.read_text(encoding="utf-8"))
    finally:
        if connection is not None:
            connection.close()
        listener.close()
        socket_path.unlink(missing_ok=True)
