"""Coordinator-owned history. Only explicit public fields cross the worker boundary."""

from collections import deque
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image

from latency_bench.core.types import Observation


class PublicHistory:
    """Save legal images, frozen request snapshots and observable execution events."""

    def __init__(self, output_dir, *, clock, image_offsets, action_history, tools=None, request_policy=None):
        self.output_dir = Path(output_dir)
        self.clock = clock
        self.image_offsets = image_offsets
        self.action_history = action_history
        self.tools = tools
        self.request_policy = request_policy

    def start_episode(self, episode_id):
        self.directory = self.output_dir / f"episode_{episode_id:03d}"
        self.directory.mkdir(parents=True)
        (self.directory / "images").mkdir()
        self.episode_id = episode_id
        self.version = 0
        self.images = deque(maxlen=max(self.image_offsets) + 1)
        self.events = deque(maxlen=32)
        self.executed = deque(maxlen=self.action_history)
        self.in_flight = {}
        self.snapshots = {}
        self.current_plan = None
        self.memory = None
        self.tool_results = []

    def record(self, kind, frame, **fields):
        self.version += 1
        event = {"version": self.version, "kind": kind, "frame": frame,
                 "control_time_ms": self.clock.step_to_time_ms(frame), **fields}
        self.events.append(event)
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def observe(self, obs_id, frame, image):
        path = self.directory / "images" / f"{obs_id:06d}.png"
        Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR).save(path)
        item = {"obs_id": obs_id, "frame": frame,
                "control_time_ms": self.clock.step_to_time_ms(frame), "path": str(path.resolve())}
        self.images.append(item)
        self.record("observation", frame, observation=item)

    def snapshot(self, obs_id, frame):
        """Build public input without registering a request or consuming latency."""
        indices = sorted({max(0, len(self.images) - 1 - offset) for offset in self.image_offsets})
        selected = [self.images[index] for index in indices]
        plan = deepcopy(self.current_plan)
        if plan is not None:
            start = plan["source_frame"] if plan["alignment"] == "observation" else plan["accepted_frame"]
            plan["elapsed_frames"] = frame - start
        snapshot = {"version": self.version, "episode_id": self.episode_id, "obs_id": obs_id,
                    "frame": frame, "control_time_ms": self.clock.step_to_time_ms(frame),
                    "images": list(selected), "executed_actions": list(self.executed),
                    "current_plan": plan, "events": list(self.events),
                    "in_flight": [{"request_id": request_id, "source_frame": source,
                                   "waited_ms": self.clock.step_to_time_ms(frame - source)}
                                  for request_id, source in self.in_flight.items()],
                    "memory": deepcopy(self.memory), "tool_results": deepcopy(self.tool_results)}
        return deepcopy(snapshot)

    def should_request(self, obs_id, frame):
        self.poll_tools(frame)
        if self.request_policy is None:
            return True
        allowed = self.request_policy(self.snapshot(obs_id, frame), {})
        if type(allowed) is not bool:
            raise ValueError("Request policy must return a boolean")
        if not allowed:
            self.record("request_skipped", frame, obs_id=obs_id)
        return allowed

    def request(self, obs_id, frame):
        self.poll_tools(frame)
        snapshot = self.snapshot(obs_id, frame)
        self.snapshots[obs_id] = snapshot
        self.in_flight[obs_id] = frame
        (self.directory / f"request_{obs_id:06d}.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        self.record("request", frame, request_id=obs_id, snapshot_version=snapshot["version"])
        return Observation(data=None, env_step=frame, sim_time_ms=snapshot["control_time_ms"],
                           metadata={"episode_id": self.episode_id, "obs_id": obs_id, "harness": snapshot})

    def returned(self, event, frame):
        request_id = event.latency.obs_id
        source = self.in_flight.pop(request_id)
        snapshot = self.snapshots.pop(request_id)
        self.record("returned", frame, request_id=request_id, source_frame=source,
                    action=event.action.value, plan=event.metadata.get("harness_plan"),
                    response=event.metadata.get("harness_response"),
                    real_worker_id=event.metadata.get("real_worker_id"),
                    virtual_worker_slot=event.latency.metadata.get("worker_id"))
        if "harness_memory" in event.metadata:
            self.memory = {"source_frame": source, "returned_frame": frame,
                           "request_id": request_id, "value": event.metadata["harness_memory"]}
            self.record("memory", frame, memory=self.memory)
        for request in event.metadata.get("harness_tools", []):
            task_id = self.tools.submit(request, snapshot, frame)
            self.record("tool_request", frame, task_id=task_id, request_id=request_id, request=request)
        if event.dropped:
            self.record("discarded", frame, request_id=request_id, reason=event.metadata["discard_reason"])

    def poll_tools(self, frame):
        if self.tools is not None:
            for result in self.tools.poll(frame):
                self.tool_results.append(result)
                self.record("tool_result", frame, **result)

    def accepted(self, event, frame, action):
        if event is None:
            return
        request_id = event.latency.obs_id
        if event.apply_raw_frame == frame and (self.current_plan is None or self.current_plan["request_id"] != request_id):
            if self.current_plan is not None:
                self.record("replaced", frame, request_id=self.current_plan["request_id"], by=request_id)
            plan = event.metadata.get("harness_plan")
            self.current_plan = {"request_id": request_id, "source_frame": event.obs_env_step,
                                 "accepted_frame": frame, "alignment": "acceptance",
                                 "action": action.value, "end": "hold", "actions": None}
            if plan is not None:
                self.current_plan.update(plan)
            self.record("accepted", frame, **self.current_plan)
            self.executed.append({"frame": frame, "action": action.value, "request_id": request_id})
        self.current_plan["action"] = action.value

    def executed_action(self, frame, action, event, reward):
        self.accepted(event, frame, action)
        request_id = None if event is None else event.latency.obs_id
        executed = {"start_frame": frame, "end_frame": frame + 1, "action": action.value,
                    "request_id": request_id, "reward": reward}
        self.record("executed", frame, **executed)

    def finish(self, frame, *, interrupted=False):
        reason = "episode_interrupted" if interrupted else "episode_ended_before_arrival"
        for request_id in self.in_flight:
            self.record("discarded", frame, request_id=request_id, reason=reason)
        self.in_flight.clear()
        self.snapshots.clear()
        if self.tools is not None:
            for task_id in self.tools.discard_pending():
                self.record("tool_discarded", frame, task_id=task_id,
                            reason="episode_interrupted" if interrupted else "episode_ended_before_result")
        self.record("episode_end", frame, status="interrupted" if interrupted else "completed")

    def close(self):
        if self.tools is not None:
            self.tools.close()
