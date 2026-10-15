"""Stateless harness decisions over the existing Codex JSON-RPC transport."""

import json
import io
import math
from pathlib import Path
import shutil
import subprocess
import time

from latency_bench.harness.protocol import decode_response
from latency_bench.harness.candidate import CandidateRuntime
from latency_bench.core.config import load_factory
from latency_bench.policy.codex import CodexPolicyRunner
from latency_bench.policy.coding_agent import InvalidAgentAction


class HarnessCodexRunner(CodexPolicyRunner):
    """Keep one app-server per worker and create one thread per public snapshot."""

    def __init__(self, config, action_resolver):
        self.config = config
        self.settings = config["policy"]
        self.actions = {int(action.value): action for action in action_resolver.resolve_many(self.settings["actions"])}
        self.instructions = Path(self.settings["model_instructions_file"]).read_text()
        self.prompt = Path(self.settings["prompt_file"]).read_text()
        self.candidate = (CandidateRuntime(self.settings["candidate_root"], self.settings["bubblewrap_executable"])
                          if "candidate_root" in self.settings else None)
        self.format_prompt = (None if self.candidate is not None else load_factory(self.settings["prompt_factory"]))
        self.output = Path(config["logging"]["output_dir"]) / "agent" / f"worker_{self.settings['worker_id']}"
        self.output.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.runtime = None

    def _archive_home(self, destination):
        # The subscription login directory is shared and remains owned by Codex.
        pass

    def _sandbox_command(self, workspace, runtime):
        (workspace / "worker_instructions.txt").write_text(self.instructions)
        return super()._sandbox_command(workspace, runtime)

    def _codex_settings(self):
        return {**super()._codex_settings(),
                "model_instructions_file": "/workspace/worker_instructions.txt",
                "forced_login_method": "chatgpt", "web_search": "disabled",
                "model_provider": "openai",
                "developer_instructions": "",
                "include_permissions_instructions": False,
                "tools.view_image": False,
                "features.shell_tool": False, "features.shell_snapshot": False,
                "features.memories": False, "features.skill_search": False,
                "features.skip_host_skill_discovery": True, "features.sleep_tool": False,
                "features.code_mode": False, "features.code_mode_host": False, "features.tool_suggest": False,
                "features.multi_agent": False}

    def _initialize_session(self):
        self.rpc_id = 0
        self.thread_id = None
        deadline = time.monotonic() + self.settings["turn_timeout_seconds"]
        with (self.episode_dir / "session_events.jsonl").open("w") as events:
            self._rpc("initialize", {"clientInfo": {"name": "latency_bench_harness", "version": "0.1.0"}},
                      events, deadline, self.episode_dir / "initialize_request.json")
            self._send({"method": "initialized", "params": {}})
            with io.StringIO() as account_events:
                account, _ = self._rpc("account/read", {"refreshToken": False}, account_events, deadline,
                                       self.episode_dir / "account_request.json")
            if account["account"] is None or account["account"]["type"] != "chatgpt":
                raise ValueError("Harness workers require ChatGPT subscription authentication")
            config, _ = self._rpc("config/read", {"includeLayers": False}, events, deadline,
                                  self.episode_dir / "config_request.json")
            if config["config"]["model_instructions_file"] != "/workspace/worker_instructions.txt":
                raise ValueError("Codex did not select the worker instruction file")
        (self.episode_dir / "effective_config.json").write_text(json.dumps(config, indent=2) + "\n")
        (self.episode_dir / "worker_instructions.txt").write_text(self.instructions)
        version = subprocess.check_output([str(Path(self.settings["package_root"]) / "bin/codex"), "--version"], text=True)
        (self.episode_dir / "cli_version.txt").write_text(version)

    def _new_thread(self, call_dir):
        deadline = time.monotonic() + self.settings["turn_timeout_seconds"]
        with (call_dir / "thread_events.jsonl").open("w") as events:
            result, _ = self._rpc("thread/start", {
                "model": self.settings["model"], "cwd": "/workspace", "approvalPolicy": "never",
                "sandbox": "read-only", "ephemeral": True, "baseInstructions": self.instructions,
            }, events, deadline, call_dir / "thread_request.json")
        (call_dir / "thread.json").write_text(json.dumps(result, indent=2) + "\n")
        if result["model"] != self.settings["model"] or result["reasoningEffort"] != self.settings["effort"]:
            raise ValueError("Codex did not select the requested model and effort")
        if result["instructionSources"]:
            raise ValueError("Codex loaded additional instruction files")
        self.thread_id = result["thread"]["id"]

    def _check_limits(self, call_dir):
        deadline = time.monotonic() + self.settings["turn_timeout_seconds"]
        with io.StringIO() as events:
            result, _ = self._rpc("account/rateLimits/read", {}, events, deadline, call_dir / "limits_request.json")
        summary = check_limits(result, self.settings["stop_used_percent"])
        (call_dir / "limits.json").write_text(json.dumps(summary, indent=2) + "\n")

    def predict(self, observation):
        if self.process is None:
            self._start_episode(observation.metadata["episode_id"])
        call_dir = self.episode_dir / f"decision_{self.decision_id:05d}"
        call_dir.mkdir()
        started = time.perf_counter()
        record = {"obs_id": observation.metadata["obs_id"], "frame": observation.env_step,
                  "model": self.settings["model"], "effort": self.settings["effort"]}
        try:
            self._check_limits(call_dir)
            self._new_thread(call_dir)
            snapshot = observation.metadata["harness"]
            images = []
            public = {**snapshot, "images": []}
            for index, item in enumerate(snapshot["images"]):
                name = f"observations/{self.decision_id:05d}_{index}.png"
                shutil.copyfile(item["path"], self.workspace / name)
                shutil.copyfile(item["path"], call_dir / f"image_{index}.png")
                path = "/workspace/" + name
                public["images"].append({**item, "path": path})
                images.append({"path": path})
            prompt = (self.candidate.call(self.settings["prompt_factory"], "prompt", snapshot, {"base": self.prompt})
                      if self.candidate is not None else self.format_prompt(self.prompt, public))
            if not isinstance(prompt, str):
                raise ValueError("Prompt formatter must return text")
            (call_dir / "snapshot.json").write_text(json.dumps(public, indent=2) + "\n")
            answer, result = self._request_action(prompt, images, call_dir)
            record.update(answer=answer, result=result, thread_id=self.thread_id)
            try:
                output = decode_response(answer, self.actions, plans_enabled=self.config["harness"]["plans_enabled"],
                                         tools_enabled=bool(self.config["harness"]["tools"]),
                                         memory_enabled=self.config["harness"]["memory_enabled"])
            except (ValueError, KeyError, TypeError) as error:
                raise InvalidAgentAction(str(error)) from error
            record["status"] = "completed"
            output.metadata.update(agent_wall_ms=(time.perf_counter() - started) * 1000,
                                   agent_call_dir=str(call_dir), agent_thread_id=self.thread_id,
                                   real_worker_id=self.settings["worker_id"], harness_response=answer)
            self.decision_id += 1
            return output
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            record["wall_ms"] = (time.perf_counter() - started) * 1000
            (call_dir / "call.json").write_text(json.dumps(record, indent=2) + "\n")


def build_policy(config, action_resolver):
    return HarnessCodexRunner(config, action_resolver)


def check_limits(result, stop_percent):
    """Enforce ordinary subscription headroom. Credits do not authorize a turn."""
    limits = result["rateLimits"]
    if result["ordinaryUsageAllowed"] is not True or limits["spendControlReached"] or limits["rateLimitReachedType"]:
        raise RuntimeError("Ordinary ChatGPT subscription usage is unavailable")
    windows = []
    for name in ("primary", "secondary"):
        window = limits[name]
        if window is None:
            continue
        used = window["usedPercent"]
        if type(used) not in (int, float) or not math.isfinite(used) or not 0 <= used <= 100:
            raise ValueError("Invalid ChatGPT subscription usage")
        if stop_percent is not None and used >= stop_percent:
            raise RuntimeError(f"ChatGPT subscription usage {used}% reached the stop line {stop_percent}%")
        windows.append({"name": name, "used_percent": used, "window_minutes": window["windowDurationMins"]})
    if not windows:
        raise RuntimeError("ChatGPT subscription usage is unavailable")
    return {"windows": windows, "stop_used_percent": stop_percent}
