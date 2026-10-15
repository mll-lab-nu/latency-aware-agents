"""Codex's native app-server protocol, one ephemeral thread per episode."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import shutil
import time

from latency_bench.policy.coding_agent import CodingAgentPolicyRunner, sandbox_command


class CodexPolicyRunner(CodingAgentPolicyRunner):
    provider = "codex"

    def _archive_home(self, destination: Path) -> None:
        destination.mkdir()
        # This dedicated home belongs only to this experiment. Auth stays in place
        # so native token rotation persists across fresh episode processes.
        for path in Path(self.settings["codex_home"]).iterdir():
            if path.name != "auth.json":
                shutil.move(str(path), destination / path.name)

    def _sandbox_command(self, workspace, runtime):
        self._archive_home(self.episode_dir / "prior_native_state")
        mounts = [
            "--ro-bind", self.settings["package_root"], "/opt/codex",
            "--bind", self.settings["codex_home"], "/home/ubuntu/.codex",
            "--bind", str(runtime), "/runtime",
        ]
        environment = {
            "PATH": "/opt/codex/bin:/usr/local/bin:/usr/bin:/bin",
            "CODEX_HOME": "/home/ubuntu/.codex",
        }
        command = ["/opt/codex/bin/codex", "app-server", "--stdio"]
        for key, value in self._codex_settings().items():
            command.extend(["-c", f"{key}={json.dumps(value)}"])
        return sandbox_command(self.settings["bubblewrap_executable"], workspace, mounts, environment, command)

    def _codex_settings(self):
        return {
            "cli_auth_credentials_store": "file",
            "model": self.settings["model"],
            "model_reasoning_effort": self.settings["effort"],
            "features.apps": False,
            "features.plugins": False,
            "memories.use_memories": False,
            "memories.generate_memories": False,
            "project_doc_max_bytes": 0,
            "history.persistence": "none",
            "log_dir": "/runtime/log",
            "sqlite_home": "/runtime/state",
        }

    def _send(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

    def _rpc(self, method, params, events, deadline, request_path):
        request = {"id": self.rpc_id, "method": method, "params": params}
        self.rpc_id += 1
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        self._send(request)
        pending = deque()
        while True:
            event = self._receive_event(deadline, events)
            if "id" in event and event["id"] == request["id"] and "method" not in event:
                if "error" in event:
                    raise RuntimeError(f"Codex {method} failed: {event['error']}")
                return event["result"], pending
            if "id" in event and "method" in event:
                raise RuntimeError(f"Unexpected interactive Codex request: {event['method']}")
            pending.append(event)

    def _initialize_session(self):
        self.rpc_id = 0
        deadline = time.monotonic() + self.settings["turn_timeout_seconds"]
        with (self.episode_dir / "session_events.jsonl").open("w") as events:
            self._rpc("initialize", {
                "clientInfo": {"name": "latency_bench", "version": "0.1.0"},
            }, events, deadline, self.episode_dir / "initialize_request.json")
            self._send({"method": "initialized", "params": {}})
            result, _ = self._rpc("thread/start", {
                "model": self.settings["model"], "cwd": "/workspace",
                "approvalPolicy": "never", "sandbox": "danger-full-access",
                "ephemeral": True,
            }, events, deadline, self.episode_dir / "thread_request.json")
        (self.episode_dir / "thread.json").write_text(json.dumps(result, indent=2) + "\n")
        if result["model"] != self.settings["model"] or result["reasoningEffort"] != self.settings["effort"]:
            raise ValueError("Codex did not select the requested model and effort")
        if result["instructionSources"]:
            raise ValueError("Codex loaded additional instruction files")
        self.thread_id = result["thread"]["id"]
        self.session_id = self.thread_id

    def _request_action(self, prompt, images, call_dir):
        deadline = time.monotonic() + self.settings["turn_timeout_seconds"]
        messages = []
        usage = None
        with (call_dir / "events.jsonl").open("w") as events:
            result, pending = self._rpc("turn/start", {
                "threadId": self.thread_id,
                "model": self.settings["model"], "effort": self.settings["effort"],
                "approvalPolicy": "never",
                "sandboxPolicy": {"type": "externalSandbox", "networkAccess": "enabled"},
                "input": [
                    {"type": "text", "text": prompt},
                    *[{"type": "localImage", "path": image["path"]} for image in images],
                ],
            }, events, deadline, call_dir / "input.json")
            turn_id = result["turn"]["id"]
            while True:
                event = pending.popleft() if pending else self._receive_event(deadline, events)
                if "id" in event:
                    raise RuntimeError(f"Unexpected interactive Codex request: {event}")
                method = event["method"]
                params = event["params"]
                if "threadId" in params and params["threadId"] != self.thread_id:
                    continue
                if "turnId" in params and params["turnId"] != turn_id:
                    continue
                if method == "item/completed" and params["item"]["type"] == "agentMessage":
                    messages.append(params["item"]["text"])
                elif method == "thread/tokenUsage/updated":
                    usage = params["tokenUsage"]
                elif method == "model/rerouted":
                    raise ValueError(f"Codex changed the requested model: {params}")
                elif method == "turn/completed":
                    if params["threadId"] != self.thread_id or params["turn"]["id"] != turn_id:
                        raise ValueError("Codex completed a different thread or turn")
                    if params["turn"]["status"] != "completed":
                        raise RuntimeError(f"Codex turn failed: {params['turn']}")
                    return messages[-1], {"turn": params["turn"], "tokenUsage": usage}

    def _finish_session(self):
        self._archive_home(self.episode_dir / "native_state")


def build_codex_policy(config, action_resolver):
    return CodexPolicyRunner(config, action_resolver)
