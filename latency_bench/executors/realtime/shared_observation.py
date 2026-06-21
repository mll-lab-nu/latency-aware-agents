from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from latency_bench.core.types import Observation


@dataclass(frozen=True)
class ObservationSlotRef:
    name: str
    slot_id: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _SlotArray:
    name: str
    shape: tuple[int, ...]
    dtype: str
    block: shared_memory.SharedMemory


@dataclass
class AttachedSharedObservation:
    observation: Observation
    blocks: list[shared_memory.SharedMemory]

    def close(self) -> None:
        for block in self.blocks:
            block.close()


@dataclass
class ObservationAttachmentCache:
    blocks: dict[str, shared_memory.SharedMemory]

    def attach(self, observation: Observation) -> AttachedSharedObservation:
        data = _attach_value(observation.data, [], cache=self)
        metadata = {
            key: _attach_value(value, [], cache=self)
            for key, value in observation.metadata.items()
        }
        return AttachedSharedObservation(
            Observation(
                data=data,
                env_step=observation.env_step,
                sim_time_ms=observation.sim_time_ms,
                metadata=metadata,
            ),
            [],
        )

    def block(self, name: str) -> shared_memory.SharedMemory:
        if name not in self.blocks:
            self.blocks[name] = shared_memory.SharedMemory(name=name)
        return self.blocks[name]

    def close(self) -> None:
        for block in self.blocks.values():
            block.close()


@dataclass
class ObservationSlotPool:
    """Fixed-capacity pool of shared-memory slots for handing observations to a
    worker process. Blocks are allocated once (sized ``nbytes * capacity``) and
    slots are reused via a free list. This is NOT a ring buffer: it does not wrap
    or overwrite the oldest entry — ``write`` raises if all slots are in flight.
    Callers must keep the number of un-``release``-d observations <= capacity.
    """

    capacity: int
    blocks: list[shared_memory.SharedMemory]
    data_template: Any
    metadata_template: dict[str, Any]
    free_slots: list[int]
    obs_slots: dict[int, int]

    @classmethod
    def from_observation(cls, observation: Observation, *, capacity: int) -> "ObservationSlotPool":
        blocks: list[shared_memory.SharedMemory] = []
        data_template = _allocate_slot_value(observation.data, blocks, capacity)
        metadata_template = {
            key: _allocate_slot_value(value, blocks, capacity)
            for key, value in observation.metadata.items()
        }
        return cls(
            capacity=capacity,
            blocks=blocks,
            data_template=data_template,
            metadata_template=metadata_template,
            free_slots=list(range(capacity)),
            obs_slots={},
        )

    def write(self, *, obs_id: int, observation: Observation) -> Observation:
        slot_id = self.free_slots.pop(0)
        self.obs_slots[int(obs_id)] = slot_id
        data = _write_slot_value(observation.data, self.data_template, slot_id)
        metadata = {
            key: _write_slot_value(observation.metadata[key], template, slot_id)
            for key, template in self.metadata_template.items()
        }
        return Observation(
            data=data,
            env_step=observation.env_step,
            sim_time_ms=observation.sim_time_ms,
            metadata=metadata,
        )

    def release(self, obs_id: int) -> None:
        slot_id = self.obs_slots.pop(int(obs_id))
        self.free_slots.insert(0, slot_id)

    def close(self) -> None:
        for block in self.blocks:
            block.close()
            block.unlink()


class ObservationLocalizer:
    def __init__(self):
        self.buffers: dict[tuple[Any, ...], np.ndarray] = {}

    def localize(self, observation: Observation) -> Observation:
        """Copy each shared array into a reused worker-local contiguous buffer so
        the policy gets a private, mutable copy and never mutates the shared slot.

        Buffers are reused across calls, so the returned observation is only valid
        until the next ``localize()`` call (the worker localizes then predicts
        synchronously, so this is safe in that flow).
        """
        return Observation(
            data=self._localize_value(observation.data, ("data",)),
            env_step=observation.env_step,
            sim_time_ms=observation.sim_time_ms,
            metadata={
                key: self._localize_value(value, ("metadata", key))
                for key, value in observation.metadata.items()
            },
        )

    def _localize_value(self, value: Any, path: tuple[Any, ...]) -> Any:
        if isinstance(value, np.ndarray):
            key = (path, value.shape, value.dtype.str)
            if key not in self.buffers:
                self.buffers[key] = np.empty(value.shape, dtype=value.dtype, order="C")
            np.copyto(self.buffers[key], value)
            return self.buffers[key]
        if isinstance(value, dict):
            return {
                key: self._localize_value(item, (*path, key))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._localize_value(item, (*path, index))
                for index, item in enumerate(value)
            ]
        if isinstance(value, tuple):
            return tuple(
                self._localize_value(item, (*path, index))
                for index, item in enumerate(value)
            )
        return value


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


def _allocate_slot_value(value: Any, blocks: list[shared_memory.SharedMemory], capacity: int) -> Any:
    if isinstance(value, np.ndarray):
        block = shared_memory.SharedMemory(create=True, size=value.nbytes * capacity)
        blocks.append(block)
        return _SlotArray(
            name=block.name,
            shape=tuple(value.shape),
            dtype=value.dtype.str,
            block=block,
        )
    if isinstance(value, dict):
        return {
            key: _allocate_slot_value(item, blocks, capacity)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _allocate_slot_value(item, blocks, capacity)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _allocate_slot_value(item, blocks, capacity)
            for item in value
        )
    return value


def _write_slot_value(value: Any, template: Any, slot_id: int) -> Any:
    if isinstance(template, _SlotArray):
        dtype = np.dtype(template.dtype)
        slot_nbytes = int(np.prod(template.shape)) * dtype.itemsize
        slot = np.ndarray(
            template.shape,
            dtype=dtype,
            buffer=template.block.buf,
            offset=int(slot_id) * slot_nbytes,
        )
        slot[...] = value
        return ObservationSlotRef(
            name=template.name,
            slot_id=int(slot_id),
            shape=template.shape,
            dtype=template.dtype,
        )
    if isinstance(template, dict):
        return {
            key: _write_slot_value(value[key], item, slot_id)
            for key, item in template.items()
        }
    if isinstance(template, list):
        return [
            _write_slot_value(value[index], item, slot_id)
            for index, item in enumerate(template)
        ]
    if isinstance(template, tuple):
        return tuple(
            _write_slot_value(value[index], item, slot_id)
            for index, item in enumerate(template)
        )
    return value


def _attach_value(
    value: Any,
    blocks: list[shared_memory.SharedMemory],
    *,
    cache: ObservationAttachmentCache | None = None,
) -> Any:
    if isinstance(value, ObservationSlotRef):
        block = cache.block(value.name) if cache is not None else shared_memory.SharedMemory(name=value.name)
        if cache is None:
            blocks.append(block)
        dtype = np.dtype(value.dtype)
        slot_nbytes = int(np.prod(value.shape)) * dtype.itemsize
        return np.ndarray(
            value.shape,
            dtype=dtype,
            buffer=block.buf,
            offset=int(value.slot_id) * slot_nbytes,
        )
    if isinstance(value, dict):
        return {
            key: _attach_value(item, blocks, cache=cache)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _attach_value(item, blocks, cache=cache)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _attach_value(item, blocks, cache=cache)
            for item in value
        )
    return value
