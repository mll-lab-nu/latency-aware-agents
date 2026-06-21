from __future__ import annotations

import numpy as np
import pytest

from latency_bench.core.types import Observation
from latency_bench.executors.realtime.shared_observation import (
    ObservationLocalizer,
    ObservationSlotPool,
    ObservationSlotRef,
    attach_observation_arrays,
)


def test_observation_slot_pool_reuses_slots_without_reallocating_blocks():
    observation = Observation(
        data=np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        env_step=7,
        sim_time_ms=123.0,
        metadata={
            "env_raw_rgb_frame_stack": np.arange(24, dtype=np.uint8).reshape(1, 2, 4, 3),
            "score": 4,
        },
    )
    owner = ObservationSlotPool.from_observation(observation, capacity=2)
    first_block_names = [block.name for block in owner.blocks]

    try:
        first = owner.write(obs_id=10, observation=observation)
        owner.release(10)
        second = owner.write(obs_id=11, observation=observation)

        assert isinstance(first.data, ObservationSlotRef)
        assert isinstance(first.metadata["env_raw_rgb_frame_stack"], ObservationSlotRef)
        assert [block.name for block in owner.blocks] == first_block_names
        assert second.data.slot_id == first.data.slot_id
        assert second.metadata["env_raw_rgb_frame_stack"].slot_id == first.metadata["env_raw_rgb_frame_stack"].slot_id
    finally:
        owner.close()

    with pytest.raises(FileNotFoundError):
        attach_observation_arrays(first)


def test_observation_localizer_returns_worker_local_contiguous_arrays():
    observation = Observation(
        data=np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        env_step=7,
        sim_time_ms=123.0,
        metadata={
            "env_raw_rgb_frame_stack": np.arange(24, dtype=np.uint8).reshape(1, 2, 4, 3),
            "score": 4,
        },
    )
    owner = ObservationSlotPool.from_observation(observation, capacity=1)
    localizer = ObservationLocalizer()

    try:
        shared = owner.write(obs_id=10, observation=observation)
        attached = attach_observation_arrays(shared)
        try:
            localized = localizer.localize(attached.observation)
        finally:
            attached.close()

        assert localized.env_step == observation.env_step
        assert localized.sim_time_ms == observation.sim_time_ms
        assert localized.metadata["score"] == 4
        assert localized.data.flags.c_contiguous
        assert localized.metadata["env_raw_rgb_frame_stack"].flags.c_contiguous
        assert not np.shares_memory(localized.data, observation.data)
        assert not np.shares_memory(
            localized.metadata["env_raw_rgb_frame_stack"],
            observation.metadata["env_raw_rgb_frame_stack"],
        )
        np.testing.assert_array_equal(localized.data, observation.data)
        np.testing.assert_array_equal(
            localized.metadata["env_raw_rgb_frame_stack"],
            observation.metadata["env_raw_rgb_frame_stack"],
        )
    finally:
        owner.close()
