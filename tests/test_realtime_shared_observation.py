from __future__ import annotations

import numpy as np
import pytest

from latency_bench.core.types import Observation
from latency_bench.executors.realtime.shared_observation import (
    SharedArrayRef,
    attach_observation_arrays,
    share_observation_arrays,
)


def test_shared_observation_roundtrip_preserves_numpy_arrays_and_scalars():
    observation = Observation(
        data=np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        env_step=7,
        sim_time_ms=123.0,
        metadata={
            "env_raw_rgb_frame_stack": np.arange(24, dtype=np.uint8).reshape(1, 2, 4, 3),
            "score": 4,
        },
    )

    shared_observation, owner = share_observation_arrays(observation)
    attached = attach_observation_arrays(shared_observation)
    try:
        assert isinstance(shared_observation.data, SharedArrayRef)
        assert isinstance(shared_observation.metadata["env_raw_rgb_frame_stack"], SharedArrayRef)
        assert attached.observation.env_step == observation.env_step
        assert attached.observation.sim_time_ms == observation.sim_time_ms
        assert attached.observation.metadata["score"] == 4
        np.testing.assert_array_equal(attached.observation.data, observation.data)
        np.testing.assert_array_equal(
            attached.observation.metadata["env_raw_rgb_frame_stack"],
            observation.metadata["env_raw_rgb_frame_stack"],
        )
    finally:
        attached.close()
        owner.release()


def test_shared_observation_release_unlinks_blocks():
    observation = Observation(
        data=np.arange(4, dtype=np.uint8),
        env_step=0,
        sim_time_ms=0.0,
    )

    shared_observation, owner = share_observation_arrays(observation)
    owner.release()

    with pytest.raises(FileNotFoundError):
        attach_observation_arrays(shared_observation)
