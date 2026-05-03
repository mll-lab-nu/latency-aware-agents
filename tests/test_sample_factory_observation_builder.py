from __future__ import annotations

import numpy as np
import pytest

from latency_bench.core.types import Observation
from latency_bench.observation.history import HistoryBuffer


def _builder(*, frame_stack: int = 4, base_channels: int = 3):
    from latency_bench.observation.sample_factory import SampleFactoryFrameStackObservationBuilder

    return SampleFactoryFrameStackObservationBuilder(
        frame_stack=frame_stack,
        base_channels=base_channels,
    )


def _obs(data, env_step: int = 0) -> Observation:
    return Observation(data=data, env_step=env_step, sim_time_ms=float(env_step))


def _history(*observations: Observation) -> HistoryBuffer:
    history = HistoryBuffer(max_len=max(1, len(observations)))
    for observation in observations:
        history.append(observation)
    return history


def test_chw_single_frame_stack_outputs_channel_stacked_shape():
    frames = [np.full((3, 5, 7), value, dtype=np.uint8) for value in range(4)]
    history = _history(*[_obs(frame, env_step=index) for index, frame in enumerate(frames)])

    observation = _builder().build(env=None, history=history)

    assert observation.data.shape == (12, 5, 7)


def test_reset_padding_repeats_first_frame():
    frame = np.full((3, 2, 3), 9, dtype=np.uint8)
    history = _history(_obs(frame))

    observation = _builder().build(env=None, history=history)

    expected = np.concatenate([frame, frame, frame, frame], axis=0)
    np.testing.assert_array_equal(observation.data, expected)


def test_partial_history_left_pads_with_oldest_frame():
    frame0 = np.full((3, 2, 3), 1, dtype=np.uint8)
    frame1 = np.full((3, 2, 3), 2, dtype=np.uint8)
    history = _history(_obs(frame0), _obs(frame1, env_step=1))

    observation = _builder().build(env=None, history=history)

    expected = np.concatenate([frame0, frame0, frame0, frame1], axis=0)
    np.testing.assert_array_equal(observation.data, expected)


def test_multiple_frames_concatenate_along_channel_axis():
    frames = [np.full((3, 2, 3), value, dtype=np.uint8) for value in range(4)]
    history = _history(*[_obs(frame, env_step=index) for index, frame in enumerate(frames)])

    observation = _builder().build(env=None, history=history)

    expected = np.concatenate(frames, axis=0)
    np.testing.assert_array_equal(observation.data, expected)


def test_hwc_rgb_converts_to_chw_before_stacking():
    frames = [np.full((2, 3, 3), value, dtype=np.uint8) for value in range(4)]
    history = _history(*[_obs(frame, env_step=index) for index, frame in enumerate(frames)])

    observation = _builder().build(env=None, history=history)

    expected = np.concatenate([np.transpose(frame, (2, 0, 1)) for frame in frames], axis=0)
    assert observation.data.shape == (12, 2, 3)
    np.testing.assert_array_equal(observation.data, expected)


def test_dtype_is_preserved():
    frames = [np.full((3, 2, 3), value, dtype=np.uint8) for value in range(4)]
    history = _history(*[_obs(frame, env_step=index) for index, frame in enumerate(frames)])

    observation = _builder().build(env=None, history=history)

    assert observation.data.dtype == np.uint8


def test_uses_only_recent_frame_stack_frames():
    frames = [np.full((3, 2, 3), value, dtype=np.uint8) for value in range(5)]
    history = _history(*[_obs(frame, env_step=index) for index, frame in enumerate(frames)])

    observation = _builder().build(env=None, history=history)

    expected = np.concatenate(frames[1:], axis=0)
    np.testing.assert_array_equal(observation.data, expected)


def test_already_stacked_chw_passes_through():
    stacked = np.arange(12 * 2 * 3, dtype=np.uint8).reshape(12, 2, 3)
    history = _history(_obs(stacked))

    observation = _builder().build(env=None, history=history)

    assert observation.data.shape == (12, 2, 3)
    np.testing.assert_array_equal(observation.data, stacked)


def test_invalid_shape_raises_clear_error():
    bad_frame = np.zeros((2, 5, 7), dtype=np.uint8)
    history = _history(_obs(bad_frame))

    with pytest.raises(ValueError, match=r"shape.*base_channels.*frame_stack"):
        _builder().build(env=None, history=history)
