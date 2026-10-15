"""Long-lived harness contracts: timing, public information and plan playback."""

import json
import threading
import time

import numpy as np
import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord, Observation, OrderingPolicy, PolicyOutput, StepResult
from latency_bench.envs.base import EnvAdapter
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.harness.history import PublicHistory
from latency_bench.harness.protocol import decode_response
from latency_bench.harness.tools import ToolTasks
from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import LatencySampler
from latency_bench.policy.base import PolicyRunner
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler
from latency_bench.scheduler.persistence import HoldPolicy


CLOCK = EnvClock(env_fps=7.5, obs_fps=1.875)
ACTIONS = {value: Action(value, is_noop=value == 0) for value in range(9)}


def scheduler():
    return ActionScheduler(clock=CLOCK, noop_action=ACTIONS[0], persistence_policy=HoldPolicy(),
                           ordering_policy=OrderingPolicy.LATEST_READY)


def event(output, source=0, ready=2, request_id=0):
    return ActionEvent(request_id, output.action, source, CLOCK.step_to_time_ms(ready), float(ready),
                       LatencyRecord(request_id, CLOCK.step_to_time_ms(ready - source)),
                       ready_raw_frame=ready, metadata=output.metadata, action_chunk=output.action_chunk)


@pytest.mark.parametrize("alignment,end,expected", [
    ("observation", "hold", [2, 2, 2, 2]),
    ("observation", "noop", [2, 0, 0, 0]),
    ("acceptance", "hold", [1, 1, 2, 2]),
    ("acceptance", "noop", [1, 1, 2, 0]),
])
def test_plan_alignment_end_and_replacement(alignment, end, expected):
    output = decode_response(json.dumps({"plan": {"actions": [{"action": 1, "frames": 2}, {"action": 2, "frames": 1}],
                                                   "alignment": alignment, "end": end}}), ACTIONS,
                             plans_enabled=True, tools_enabled=False, memory_enabled=False)
    queue = scheduler()
    queue.enqueue(event(output))
    assert queue.update(0, 0).value == 0
    assert [queue.update(frame, CLOCK.step_to_time_ms(frame)).value for frame in range(2, 6)] == expected
    queue.enqueue(event(PolicyOutput(ACTIONS[3]), source=5, ready=6, request_id=1))
    assert queue.update(6, CLOCK.step_to_time_ms(6)).value == 3
    assert queue.update(7, CLOCK.step_to_time_ms(7)).value == 3


@pytest.mark.parametrize("alignment,end", [("observation", "hold"), ("acceptance", "noop")])
def test_request_reports_the_action_played_by_the_plan(tmp_path, alignment, end):
    """Retained history contract: a plan's first action is not its current action."""
    history = PublicHistory(tmp_path, clock=CLOCK, image_offsets=[0], action_history=8)
    history.start_episode(0)
    output = decode_response(json.dumps({"plan": {"actions": [{"action": 2, "frames": 1}, {"action": 3, "frames": 1}],
                                                   "alignment": alignment, "end": end}}),
                             ACTIONS, plans_enabled=True, tools_enabled=False, memory_enabled=False)
    queue = scheduler()
    queue.enqueue(event(output, ready=1))
    played = []
    for frame in range(1, 5):
        action = queue.update(frame, CLOCK.step_to_time_ms(frame))
        history.accepted(queue.latest_applied_event, frame, action)
        history.observe(frame, frame, np.zeros((8, 8, 3), dtype=np.uint8))
        snapshot = history.request(frame, frame).metadata["harness"]
        assert snapshot["current_plan"]["action"] == action.value
        history.executed_action(frame, action, queue.latest_applied_event, 0)
        played.append(action.value)
    assert played == ([3, 3, 3, 3] if alignment == "observation" else [2, 3, 0, 0])


@pytest.mark.parametrize("response", [
    '{"action": true}', '{"action": 9}', '{"action": 0, "memory": "disabled"}',
    '{"plan":{"actions":[{"action":1,"frames":0}],"alignment":"observation","end":"noop"}}',
    '{"plan":{"actions":[{"action":1,"frames":1.5}],"alignment":"acceptance","end":"hold"}}',
])
def test_external_response_rejects_illegal_actions_and_durations(response):
    with pytest.raises(ValueError):
        decode_response(response, ACTIONS, plans_enabled=True, tools_enabled=False, memory_enabled=False)


class ImageEnv(EnvAdapter):
    """Expose images and deliberately private metadata for boundary tests."""

    env_fps = 7.5
    noop_action = ACTIONS[0]

    def reset(self, seed=None):
        self.env_step = 0
        return self.observe()

    def observe(self):
        return Observation(self.render_game_frame(), self.env_step, CLOCK.step_to_time_ms(self.env_step),
                           {"internal_state": "PRIVATE_SENTINEL", "latency_ms": "FUTURE_SENTINEL"})

    def render_game_frame(self):
        return np.full((24, 32, 3), self.env_step, dtype=np.uint8)

    def step(self, action):
        self.env_step += 1
        return StepResult(None, float(action.value), self.env_step == 17)


class Delays(LatencySampler):
    """Return fixed raw-frame delays for deterministic out-of-order decisions."""

    latency_type = "fixed"

    def __init__(self):
        self.delays = iter([12, 2, 1, 1, 20])

    def sample(self, context):
        return LatencySample.from_latency_ms(CLOCK.step_to_time_ms(next(self.delays)), clock=CLOCK)


class RecordingPolicy(PolicyRunner):
    """Read only public snapshots and return native single actions."""

    def __init__(self):
        self.observations = []

    def predict(self, observation):
        self.observations.append(observation)
        return PolicyOutput(ACTIONS[len(self.observations)])


def test_executor_hides_future_results_and_metadata_and_records_actual_execution(tmp_path):
    policy = RecordingPolicy()
    history = PublicHistory(tmp_path, clock=CLOCK, image_offsets=[4, 2, 0], action_history=8)
    executor = SimulatedLatencyExecutor(env=ImageEnv(), policy=policy, decision_scheduler=ObservationCadenceDecisionScheduler(CLOCK),
                                        latency_source=Delays(), action_scheduler=scheduler(), clock=CLOCK,
                                        harness=history, simulated_worker_capacity=8)
    try:
        executor.run_episode(0, seed=2, max_env_steps=17)
    finally:
        executor.close()
    snapshots = [observation.metadata["harness"] for observation in policy.observations]
    assert [[image["frame"] for image in snapshot["images"]] for snapshot in snapshots] == [
        [0], [0, 4], [0, 8], [0, 4, 12], [0, 8, 16]]
    assert set(policy.observations[0].metadata) == {"episode_id", "obs_id", "harness"}
    serialized = json.dumps(snapshots)
    assert "SENTINEL" not in serialized
    assert "ready_raw_frame" not in serialized
    assert snapshots[1]["in_flight"] == [{"request_id": 0, "source_frame": 0, "waited_ms": CLOCK.step_to_time_ms(4)}]
    assert snapshots[1]["current_plan"] is None
    assert snapshots[2]["current_plan"]["action"] == 2
    assert snapshots[2]["executed_actions"][0]["frame"] == 6
    events = [json.loads(line) for line in (tmp_path / "episode_000/events.jsonl").read_text().splitlines()]
    assert any(item["kind"] == "discarded" and item["request_id"] == 0 and item["frame"] == 12 for item in events)
    assert not any(item["kind"] == "accepted" and item["request_id"] in (0, 4) for item in events)
    assert [item["action"] for item in events if item["kind"] == "executed"][:10] == [0] * 6 + [2] * 3 + [3]


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_interruption_closes_requests_and_preserves_completed_action_records(tmp_path, monkeypatch, failure):
    """Retained executor contract: interrupted episodes keep an auditable final state."""
    env = ImageEnv()
    original_step = env.step

    def interrupt(action):
        if env.env_step == 7:
            raise failure("stop")
        return original_step(action)

    monkeypatch.setattr(env, "step", interrupt)
    history = PublicHistory(tmp_path / "public", clock=CLOCK, image_offsets=[0], action_history=8)
    logger = TrajectoryLogger(tmp_path / "logs", save_step_records=True,
                              save_action_records=True, save_latency_records=True)
    executor = SimulatedLatencyExecutor(env=env, policy=RecordingPolicy(),
                                        decision_scheduler=ObservationCadenceDecisionScheduler(CLOCK),
                                        latency_source=Delays(), action_scheduler=scheduler(), clock=CLOCK,
                                        harness=history, simulated_worker_capacity=8, logger=logger)
    try:
        with pytest.raises(failure, match="stop"):
            executor.run_episode(0, seed=2, max_env_steps=17)
    finally:
        executor.close()
        logger.flush()
    events = [json.loads(line) for line in (tmp_path / "public/episode_000/events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "episode_end" and events[-1]["status"] == "interrupted"
    issued = {item["request_id"] for item in events if item["kind"] == "request"}
    settled = {item["request_id"] for item in events if item["kind"] in {"returned", "discarded"}}
    assert issued == settled == {0, 1}
    assert any(item["kind"] == "discarded" and item["reason"] == "episode_interrupted" for item in events)
    assert len((tmp_path / "logs/steps.jsonl").read_text().splitlines()) == 7
    assert len((tmp_path / "logs/actions.jsonl").read_text().splitlines()) == 2
    actions = [json.loads(line) for line in (tmp_path / "logs/actions.jsonl").read_text().splitlines()]
    assert actions[0]["dropped"] is True
    assert actions[0]["metadata"]["discard_reason"] == "episode_interrupted"


def test_tool_and_memory_results_enter_only_later_snapshots(tmp_path):
    def information(snapshot, arguments):
        assert snapshot["frame"] == 0
        assert snapshot["tool_results"] == []
        time.sleep(0.002)
        return {"suggestion": arguments["value"]}

    tools = ToolTasks({"inspect": information}, clock=CLOCK, mode="simulated")
    history = PublicHistory(tmp_path, clock=CLOCK, image_offsets=[0], action_history=8, tools=tools)
    history.start_episode(0)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    history.observe(0, 0, image)
    original = history.request(0, 0).metadata["harness"]
    output = decode_response('{"action":2,"memory":"row 3","tools":[{"name":"inspect","input":{"value":4}}]}',
                             ACTIONS, plans_enabled=False, tools_enabled=True, memory_enabled=True)
    history.returned(event(output), 2)
    history.observe(1, 2, image)
    same_frame = history.request(1, 2).metadata["harness"]
    assert same_frame["tool_results"] == []
    assert same_frame["memory"]["source_frame"] == 0
    history.observe(2, 4, image)
    later = history.request(2, 4).metadata["harness"]
    assert later["tool_results"][0]["result"] == {"suggestion": 4}
    assert later["tool_results"][0]["visible_frame"] == 3
    assert original["memory"] is None and original["tool_results"] == []
    history.finish(4)
    history.start_episode(1)
    history.observe(3, 0, image)
    next_episode = history.request(3, 0).metadata["harness"]
    assert next_episode["memory"] is None and next_episode["tool_results"] == []
    history.close()


def test_realtime_tools_return_without_blocking_decision():
    release = threading.Event()

    def information(snapshot, arguments):
        release.wait(2)
        return "done"

    tools = ToolTasks({"inspect": information}, clock=CLOCK, mode="realtime")
    try:
        tools.submit({"name": "inspect", "input": {}}, {"frame": 0}, 0)
        assert tools.poll(1) == []
        release.set()
    finally:
        tools.close()
