from __future__ import annotations

import numpy as np

from common import compare_step_traces, step_record


def test_compare_step_traces_reports_identical_semantic_behavior():
    baseline = [
        step_record(obs=np.asarray([[1, 2], [3, 4]], dtype=np.uint8), reward=1.0, terminated=False, truncated=False),
        step_record(obs=np.asarray([[5, 6], [7, 8]], dtype=np.uint8), reward=0.0, terminated=True, truncated=False),
    ]
    accelerated = [
        step_record(obs=np.asarray([[1, 2], [3, 4]], dtype=np.uint8), reward=1.0, terminated=False, truncated=False),
        step_record(obs=np.asarray([[5, 6], [7, 8]], dtype=np.uint8), reward=0.0, terminated=True, truncated=False),
    ]

    result = compare_step_traces(baseline, accelerated)

    assert result == {
        "obs_equal": True,
        "reward_equal": True,
        "terminated_equal": True,
        "truncated_equal": True,
        "info_equal": True,
        "behavior_changed": False,
    }


def test_compare_step_traces_detects_reward_and_observation_changes():
    baseline = [
        step_record(obs=np.asarray([1, 2, 3], dtype=np.uint8), reward=1.0, terminated=False, truncated=False),
    ]
    accelerated = [
        step_record(obs=np.asarray([1, 2, 4], dtype=np.uint8), reward=2.0, terminated=False, truncated=False),
    ]

    result = compare_step_traces(baseline, accelerated)

    assert result["obs_equal"] is False
    assert result["reward_equal"] is False
    assert result["behavior_changed"] is True


def test_step_record_requires_requested_info_fields():
    try:
        step_record(
            obs=np.asarray([1], dtype=np.uint8),
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
            info_keys=("score",),
        )
    except KeyError as exc:
        assert exc.args == ("score",)
    else:
        raise AssertionError("missing requested info field did not fail")
