"""Claude Code's native stream-json protocol, one conversation per episode."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from latency_bench.policy.coding_agent import CodingAgentPolicyRunner, sandbox_command


class ClaudeCodePolicyRunner(CodingAgentPolicyRunner):
    provider = "claude_code"

    def __init__(self, config, action_resolver):
        super().__init__(config, action_resolver)
        self.credential_copy = None

    def _persist_refreshed_credentials(self) -> None:
        """Preserve OAuth rotation before removing the isolated session home."""
        source = Path(self.settings["credentials_file"])
        refreshed = json.loads(self.credential_copy.read_text())["claudeAiOauth"]
        current = json.loads(source.read_text())
        if refreshed["expiresAt"] > current["claudeAiOauth"]["expiresAt"]:
            current["claudeAiOauth"] = refreshed
            with tempfile.NamedTemporaryFile(
                mode="w", dir=source.parent, prefix=".coding-agent-auth-", delete=False,
            ) as updated:
                updated.write(json.dumps(current) + "\n")
            os.replace(updated.name, source)


    def _sandbox_command(self, workspace, credentials):
        shutil.copyfile(self.settings["credentials_file"], credentials / ".credentials.json")
        self.credential_copy = credentials / ".credentials.json"
        self.credential_copy.chmod(0o600)
        mounts = [
            "--ro-bind", str(Path(self.settings["executable"]).resolve()), "/opt/claude",
            "--bind", str(credentials), "/home/ubuntu/.claude",
        ]
        environment = {
            "PATH": "/opt:/usr/local/bin:/usr/bin:/bin",
            "CLAUDE_CONFIG_DIR": "/home/ubuntu/.claude",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1",
        }
        command = [
            "/opt/claude", "--safe-mode", "-p",
            "--model", self.settings["model"], "--effort", self.settings["effort"],
            "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--session-id", self.session_id,
            "--permission-mode", "bypassPermissions",
            "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        ]
        return sandbox_command(self.settings["bubblewrap_executable"], workspace, mounts, environment, command)

    def _initialize_session(self):
        if self.config["executor"]["mode"] != "realtime":
            return
        request = {"type": "control_request", "request_id": "initialize",
                   "request": {"subtype": "initialize", "hooks": None}}
        (self.episode_dir / "initialize_request.json").write_text(json.dumps(request) + "\n")
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + self.settings["turn_timeout_seconds"]
        with (self.episode_dir / "session_events.jsonl").open("w") as events:
            while True:
                event = self._receive_event(deadline, events)
                if event["type"] == "control_response":
                    response = event["response"]
                    if response["request_id"] != "initialize" or response["subtype"] != "success":
                        raise RuntimeError(f"Claude Code initialization failed: {response}")
                    return

    def _request_action(self, prompt, images, call_dir):
        message = {
            "type": "user", "session_id": self.session_id, "parent_tool_use_id": None,
            "message": {"role": "user", "content": [
                {"type": "text", "text": prompt},
                *[{"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.b64encode(image["bytes"]).decode("ascii"),
                }} for image in images],
            ]},
        }
        (call_dir / "input.json").write_text(json.dumps(message) + "\n")
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + self.settings["turn_timeout_seconds"]
        with (call_dir / "events.jsonl").open("w") as events:
            while True:
                event = self._receive_event(deadline, events)
                if event["type"] == "assistant":
                    if "is_api_error_message" in event and event["is_api_error_message"]:
                        continue
                    if event["message"]["model"] != self.settings["model"]:
                        raise ValueError("Claude Code returned a different model than requested")
                if event["type"] == "result":
                    if event["is_error"]:
                        raise RuntimeError(f"Claude Code error: {event}")
                    if event["session_id"] != self.session_id:
                        raise ValueError("Claude Code returned an action for a different session")
                    self._persist_refreshed_credentials()
                    return event["result"], event

    def _finish_session(self):
        if self.credential_copy is not None:
            self._persist_refreshed_credentials()
            self.credential_copy = None


def build_claude_code_policy(config, action_resolver):
    return ClaudeCodePolicyRunner(config, action_resolver)
