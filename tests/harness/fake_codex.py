"""Protocol fixture for persistent workers with fresh decision threads."""

import json
from pathlib import Path
import sys


if "--version" in sys.argv:
    print("codex-cli fixture")
    raise SystemExit

workspace = Path(sys.argv[1])
count = 0
thread = None


def send(value):
    print(json.dumps(value), flush=True)


for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    params = request["params"]
    if method == "initialized":
        continue
    if method == "initialize":
        result = {"userAgent": "fixture"}
    elif method == "account/read":
        result = {"account": {"type": "chatgpt"}}
    elif method == "config/read":
        result = {"config": {"model_instructions_file": "/workspace/worker_instructions.txt"}}
    elif method == "account/rateLimits/read":
        result = {"ordinaryUsageAllowed": True, "rateLimits": {
            "primary": {"usedPercent": 20, "windowDurationMins": 10080}, "secondary": None,
            "spendControlReached": False, "rateLimitReachedType": None}}
    elif method == "thread/start":
        assert params["baseInstructions"] == "GAME_WORKER_INSTRUCTIONS"
        assert params["ephemeral"] is True
        thread = f"thread-{count}"
        result = {"model": "gpt-6.1-sol", "reasoningEffort": "low", "instructionSources": [],
                  "thread": {"id": thread}}
        count += 1
    elif method == "turn/start":
        assert params["threadId"] == thread
        assert len(params["input"]) == 4
        for item in params["input"][1:]:
            assert (workspace / item["path"].removeprefix("/workspace/")).exists()
        send({"id": request["id"], "result": {"turn": {"id": thread}}})
        send({"method": "thread/tokenUsage/updated", "params": {"threadId": thread, "tokenUsage": {
            "last": {"inputTokens": 120, "cachedInputTokens": 10, "outputTokens": 5}}}})
        send({"method": "item/completed", "params": {"threadId": thread, "turnId": thread,
              "item": {"type": "agentMessage", "text": str(count)}}})
        send({"method": "turn/completed", "params": {"threadId": thread,
              "turn": {"id": thread, "status": "completed", "items": []}}})
        continue
    else:
        raise ValueError(method)
    send({"id": request["id"], "result": result})
