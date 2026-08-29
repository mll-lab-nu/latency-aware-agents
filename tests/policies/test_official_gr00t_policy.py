from __future__ import annotations

import numpy as np
import pytest

from latency_bench.core.types import Observation
from latency_bench.envs.humanoidbench_sonic_contract import (
    G1_GR00T_ACTION_DIM,
    G1_GR00T_ACTION_HORIZON,
)
from latency_bench.policy.official_gr00t import (
    OFFICIAL_GR00T_ACTION_KEY,
    OFFICIAL_GR00T_PROPRIO_INFO_KEY,
    OfficialGr00tPolicyRunner,
)


class _FakeGr00tPolicy:
    def __init__(self) -> None:
        self.observations = []

    def get_action(self, observation):
        self.observations.append(observation)
        batch_size = len(observation["language"]["annotation.human.task_description"])
        action = np.arange(
            batch_size * G1_GR00T_ACTION_HORIZON * G1_GR00T_ACTION_DIM,
            dtype=np.float32,
        ).reshape(batch_size, G1_GR00T_ACTION_HORIZON, G1_GR00T_ACTION_DIM)
        return {"action": action}, {"unused": True}


def _observation() -> Observation:
    frames = np.zeros((4, 2, 3, 3), dtype=np.uint8)
    frames[-1] = 7
    return Observation(
        data=None,
        env_step=4,
        sim_time_ms=80.0,
        metadata={
            "env_raw_rgb_frame_stack": frames,
            OFFICIAL_GR00T_PROPRIO_INFO_KEY: np.arange(93, dtype=np.float32),
        },
    )


def test_official_runner_emits_nested_input_and_40_by_29_chunk():
    assert OFFICIAL_GR00T_ACTION_KEY == "action"
    fake_policy = _FakeGr00tPolicy()
    runner = OfficialGr00tPolicyRunner(
        policy=fake_policy,
        prompt="run forward",
        model_path="checkpoint",
        embodiment_tag="NEW_EMBODIMENT",
        device="cuda:0",
    )

    output = runner.predict(_observation())

    sent = fake_policy.observations[0]
    assert set(sent) == {"video", "state", "language"}
    assert sent["video"]["ego_view"].shape == (1, 1, 2, 3, 3)
    assert sent["video"]["ego_view"].dtype == np.uint8
    assert sent["video"]["ego_view"][0, 0, 0, 0, 0] == 7
    assert sent["state"]["state"].shape == (1, 1, 93)
    assert sent["state"]["state"].dtype == np.float32
    assert sent["language"]["annotation.human.task_description"] == [["run forward"]]
    assert output.action_chunk.shape == (40, 29)
    assert output.action.value.shape == (29,)
    np.testing.assert_array_equal(output.action_chunk[0], output.action.value)


def test_official_runner_requires_the_exact_proprioception_field():
    observation = _observation()
    del observation.metadata[OFFICIAL_GR00T_PROPRIO_INFO_KEY]
    runner = OfficialGr00tPolicyRunner(
        policy=_FakeGr00tPolicy(),
        prompt="run forward",
        model_path="checkpoint",
        embodiment_tag="NEW_EMBODIMENT",
        device="cpu",
    )

    with pytest.raises(KeyError, match=OFFICIAL_GR00T_PROPRIO_INFO_KEY):
        runner.predict(observation)


def test_official_runner_rejects_an_action_chunk_with_the_wrong_shape():
    class WrongShapePolicy(_FakeGr00tPolicy):
        def get_action(self, observation):
            return {OFFICIAL_GR00T_ACTION_KEY: np.zeros((1, 29), dtype=np.float32)}, {}

    runner = OfficialGr00tPolicyRunner(
        policy=WrongShapePolicy(),
        prompt="run forward",
        model_path="checkpoint",
        embodiment_tag="NEW_EMBODIMENT",
        device="cpu",
    )

    with pytest.raises(ValueError, match="40, 29"):
        runner.predict(_observation())
