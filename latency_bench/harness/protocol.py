"""Decode agent actions at the model boundary. Durations use raw frames."""

import json

from latency_bench.core.types import PolicyOutput


def decode_response(text, actions, *, plans_enabled, tools_enabled, memory_enabled):
    payload = json.loads(text)
    if type(payload) is int:
        payload = {"action": payload}
    if not isinstance(payload, dict):
        raise ValueError("The response must be a JSON object")
    allowed = {"action", "plan"} if plans_enabled else {"action"}
    if tools_enabled:
        allowed.add("tools")
    if memory_enabled:
        allowed.add("memory")
    if payload.keys() - allowed or ("action" in payload) == ("plan" in payload):
        raise ValueError("Return exactly one action or enabled plan, with only enabled optional fields")

    def action(value):
        if type(value) is not int or value not in actions:
            raise ValueError(f"Illegal action: {value!r}")
        return actions[value]

    metadata = {}
    chunk = None
    if "action" in payload:
        first = action(payload["action"])
    else:
        plan = payload["plan"]
        if not isinstance(plan, dict) or set(plan) != {"actions", "alignment", "end"}:
            raise ValueError("A plan requires actions, alignment and end")
        if plan["alignment"] not in {"observation", "acceptance"} or plan["end"] not in {"hold", "noop"}:
            raise ValueError("Invalid plan alignment or end behavior")
        if not isinstance(plan["actions"], list) or not plan["actions"]:
            raise ValueError("A plan requires at least one action")
        chunk = []
        for segment in plan["actions"]:
            if not isinstance(segment, dict) or set(segment) != {"action", "frames"}:
                raise ValueError("Each plan segment requires action and frames")
            if type(segment["frames"]) is not int or segment["frames"] <= 0:
                raise ValueError("Plan duration must be a positive integer")
            chunk.extend([action(segment["action"])] * segment["frames"])
        first = chunk[0]
        metadata["harness_plan"] = plan
    if "tools" in payload:
        if not isinstance(payload["tools"], list):
            raise ValueError("tools must be an array")
        for request in payload["tools"]:
            if not isinstance(request, dict) or set(request) != {"name", "input"} or not isinstance(request["name"], str):
                raise ValueError("Each tool request requires name and input")
        metadata["harness_tools"] = payload["tools"]
    if "memory" in payload:
        metadata["harness_memory"] = payload["memory"]
    return PolicyOutput(action=first, action_chunk=chunk, raw_output=text, metadata=metadata)
