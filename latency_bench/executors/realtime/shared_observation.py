from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from latency_bench.core.types import Observation


@dataclass(frozen=True)
class SharedArrayRef:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass
class SharedObservationOwner:
    blocks: list[shared_memory.SharedMemory]

    def release(self) -> None:
        for block in self.blocks:
            block.close()
            block.unlink()


@dataclass
class AttachedSharedObservation:
    observation: Observation
    blocks: list[shared_memory.SharedMemory]

    def close(self) -> None:
        for block in self.blocks:
            block.close()


def share_observation_arrays(observation: Observation) -> tuple[Observation, SharedObservationOwner]:
    blocks: list[shared_memory.SharedMemory] = []
    data = _share_value(observation.data, blocks)
    metadata = {
        key: _share_value(value, blocks)
        for key, value in observation.metadata.items()
    }
    shared_observation = Observation(
        data=data,
        env_step=observation.env_step,
        sim_time_ms=observation.sim_time_ms,
        metadata=metadata,
    )
    return shared_observation, SharedObservationOwner(blocks)


def attach_observation_arrays(observation: Observation) -> AttachedSharedObservation:
    blocks: list[shared_memory.SharedMemory] = []
    data = _attach_value(observation.data, blocks)
    metadata = {
        key: _attach_value(value, blocks)
        for key, value in observation.metadata.items()
    }
    attached_observation = Observation(
        data=data,
        env_step=observation.env_step,
        sim_time_ms=observation.sim_time_ms,
        metadata=metadata,
    )
    return AttachedSharedObservation(attached_observation, blocks)


def _share_value(value: Any, blocks: list[shared_memory.SharedMemory]) -> Any:
    if not isinstance(value, np.ndarray):
        return value
    block = shared_memory.SharedMemory(create=True, size=value.nbytes)
    shared_array = np.ndarray(value.shape, dtype=value.dtype, buffer=block.buf)
    shared_array[...] = value
    blocks.append(block)
    return SharedArrayRef(
        name=block.name,
        shape=tuple(value.shape),
        dtype=value.dtype.str,
    )


def _attach_value(value: Any, blocks: list[shared_memory.SharedMemory]) -> Any:
    if not isinstance(value, SharedArrayRef):
        return value
    block = shared_memory.SharedMemory(name=value.name)
    blocks.append(block)
    return np.ndarray(value.shape, dtype=np.dtype(value.dtype), buffer=block.buf)
