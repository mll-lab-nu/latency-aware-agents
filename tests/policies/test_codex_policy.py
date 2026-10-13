from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.core.types import Observation
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.policy.codex import CodexPolicyRunner


@pytest.mark.parametrize("num_images", [1, 2, 4])
def test_native_turns_keep_process_reset_context_and_preserve_auth(tmp_path, monkeypatch, num_images):
    config = load_config(Path(__file__).resolve().parents[2] / "configs/experiments/coding_agent/flappy_codex_luna_low.yaml")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"summary": {"mean_ms": 75.87, "p95_ms": 81.32}}))
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "auth.json").write_text('NATIVE_AUTH_SENTINEL')
    (home / "old-session.txt").write_text('PRIOR_EPISODE_SENTINEL')
    config["policy"].update(codex_home=str(home), workspace_root=str(tmp_path / "workspaces"), runtime_root=str(tmp_path))
    config["logging"]["output_dir"] = str(tmp_path / "output")
    config["latency"]["profile_path"] = str(profile)
    fake = tmp_path / "fake_codex.py"
    fake.write_text('''import json, sys, uuid
from pathlib import Path
home, workspace = Path(sys.argv[1]), Path(sys.argv[2])
assert sorted(p.name for p in home.iterdir()) == ["auth.json"]
thread = str(uuid.uuid4())
count = 0
def send(event):
    print(json.dumps(event), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    if method == "initialized":
        continue
    if method == "initialize":
        send({"id": request["id"], "result": {"userAgent": "test-codex"}})
    elif method == "thread/start":
        assert request["params"]["ephemeral"] is True
        assert "developerInstructions" not in request["params"]
        send({"method": "thread/started", "params": {"thread": {"id": thread}}})
        send({"id": request["id"], "result": {"model": "gpt-6-luna", "reasoningEffort": "low",
              "instructionSources": [], "thread": {"id": thread}}})
    elif method == "turn/start":
        params = request["params"]
        assert params["threadId"] == thread
        assert params["model"] == "gpt-6-luna" and params["effort"] == "low"
        assert len(params["input"]) == 1 + int(sys.argv[3])
        for index, image in enumerate(params["input"][1:]):
            suffix = "" if index == 0 else f"_{index}"
            assert image["path"] == f"/workspace/observations/{count:05d}{suffix}.png"
            assert (workspace / "observations" / f"{count:05d}{suffix}.png").exists()
        (home / "current-session.txt").write_text(thread)
        turn = str(count)
        send({"id": request["id"], "result": {"turn": {"id": turn, "status": "inProgress"}}})
        send({"method": "thread/tokenUsage/updated", "params": {"threadId": "other",
              "turnId": turn, "tokenUsage": {"last": {"totalTokens": 99999}}}})
        send({"method": "thread/tokenUsage/updated", "params": {"threadId": thread,
              "turnId": turn, "tokenUsage": {"last": {"totalTokens": 10}, "total": {"totalTokens": (count+1)*10}}}})
        send({"method": "item/completed", "params": {"threadId": thread, "turnId": turn,
              "item": {"type": "agentMessage", "text": json.dumps({"action": (count+1)%2})}}})
        send({"method": "turn/completed", "params": {"threadId": thread,
              "turn": {"id": turn, "status": "completed", "items": [], "error": None}}})
        count += 1
''')

    def launch(self, workspace, runtime):
        self._archive_home(self.episode_dir / "prior_native_state")
        return [sys.executable, str(fake), str(home), str(workspace), str(num_images)]

    monkeypatch.setattr(CodexPolicyRunner, "_sandbox_command", launch)
    policy = CodexPolicyRunner(config, ActionResolver(config["env"]))
    original_observe = policy._observation_input
    def observe(observation):
        example = original_observe(observation)
        example["image"] *= num_images
        return example
    monkeypatch.setattr(policy, "_observation_input", observe)
    observation = Observation(None, 0, 0, {
        ENV_RAW_RGB_FRAME_STACK_INFO_KEY: np.zeros((1, 32, 32, 3), dtype=np.uint8),
        "episode_id": 0, "obs_id": 0,
    })
    try:
        policy.prepare_episode(0)
        prepared_process = policy.process
        assert not (policy.episode_dir / "decision_00000").exists()
        assert policy.predict(observation).action.value == 1
        assert policy.process is prepared_process
        process = policy.process
        thread_id = policy.thread_id
        assert policy.predict(observation).action.value == 0
        assert policy.process is process
        call = json.loads((policy.episode_dir / "decision_00001/call.json").read_text())
        assert call["result"]["tokenUsage"]["last"]["totalTokens"] == 10
        assert (policy.workspace / "observations/00000.png").exists()
        assert (policy.workspace / "observations/00001.png").exists()
        policy.reset_state(0)
        assert process.poll() == 0
        assert sorted(p.name for p in home.iterdir()) == ["auth.json"]
        observation.metadata["episode_id"] = 1
        assert policy.predict(observation).action.value == 1
        assert policy.thread_id != thread_id
    finally:
        policy.close()
    assert (home / "auth.json").read_text() == 'NATIVE_AUTH_SENTINEL'
    assert sorted(p.name for p in home.iterdir()) == ["auth.json"]
