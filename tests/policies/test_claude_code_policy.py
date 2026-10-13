from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.core.types import Action, Observation
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.policy.claude_code import ClaudeCodePolicyRunner
from latency_bench.policy.coding_agent import InvalidAgentAction, decode_action
from latency_bench.policy._starvla_input import model_rgb_images


@pytest.mark.parametrize("text", [
    '{"action": true}', '{"action": 1.0}', '{"action": "1"}',
    '{"action": -1}', '{"action": 2}', '{"action": 1, "extra": 0}',
    '{}', '```json\n{"action": 1}\n```',
])
def test_invalid_external_actions_are_not_repaired(text):
    with pytest.raises((ValueError, KeyError)):
        decode_action(text, {0: Action(0), 1: Action(1)})


def test_rgb_preprocessing_matches_reference_pixels():
    frame = np.random.default_rng(0).integers(0, 256, (512, 288, 3), dtype=np.uint8)
    expected = Image.fromarray(frame).resize((224, 224))
    actual = model_rgb_images([frame], (224, 224))[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("num_images", [1, 2, 4])
@pytest.mark.parametrize("realtime", [False, True])
def test_persistent_episode_process_images_reset_and_cleanup(tmp_path, monkeypatch, num_images, realtime):
    config = load_config(Path(__file__).resolve().parents[2] / "configs/experiments/coding_agent/flappy_claude_haiku55_low.yaml")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"summary": {"mean_ms": 75.87, "p95_ms": 81.32}}))
    credentials = tmp_path / "credentials.json"
    credentials.write_text('{"claudeAiOauth": {"expiresAt": 1}}')
    config["latency"]["profile_path"] = str(profile)
    config["logging"]["output_dir"] = str(tmp_path / "output")
    config["policy"].update(
        credentials_file=str(credentials), runtime_root=str(tmp_path),
        workspace_root=str(tmp_path / "workspaces"),
    )
    if realtime:
        config["executor"]["mode"] = "realtime"
        config["latency"] = {"method": "measured"}
    fake = tmp_path / "fake_cli.py"
    fake.write_text('''import json, sys
from pathlib import Path
session = sys.argv[1]
index = 0
for line in sys.stdin:
    request = json.loads(line)
    if request["type"] == "control_request":
        assert request["request"] == {"subtype": "initialize", "hooks": None}
        print(json.dumps({"type": "control_response", "response": {
            "request_id": request["request_id"], "subtype": "success", "response": {}}}), flush=True)
        continue
    assert request["session_id"] == session
    assert request["message"]["content"][1]["type"] == "image"
    assert len(request["message"]["content"]) == 1 + int(sys.argv[3])
    assert all(image["source"]["media_type"] == "image/png" for image in request["message"]["content"][1:])
    action = (index + 1) % 2
    index += 1
    Path(sys.argv[2]).write_text('{"claudeAiOauth": {"expiresAt": 2}}')
    print(json.dumps({"type": "assistant", "message": {"model": "claude-haiku-5-5"}}), flush=True)
    print(json.dumps({"type": "result", "session_id": session, "is_error": False,
                      "result": json.dumps({"action": action})}), flush=True)
''')
    def launch(self, workspace, credentials):
        self.credential_copy = credentials / ".credentials.json"
        self.credential_copy.write_text(Path(self.settings["credentials_file"]).read_text())
        return [sys.executable, str(fake), self.session_id, str(self.credential_copy), str(num_images)]

    monkeypatch.setattr(ClaudeCodePolicyRunner, "_sandbox_command", launch)
    policy = ClaudeCodePolicyRunner(config, ActionResolver(config["env"]))
    original_observe = policy._observation_input
    def observe(observation):
        example = original_observe(observation)
        example["image"] *= num_images
        return example
    monkeypatch.setattr(policy, "_observation_input", observe)
    frame = np.random.default_rng(1).integers(0, 256, (512, 288, 3), dtype=np.uint8)
    observation = Observation(None, 0, 0, {
        ENV_RAW_RGB_FRAME_STACK_INFO_KEY: np.stack([frame]),
        "episode_id": 0, "obs_id": 0, "latency_ms": 987654321,
        "numeric_obs": "PRIVILEGED_STATE_SENTINEL",
    })
    try:
        if realtime:
            policy.prepare_episode(0)
            assert policy.process is not None
            assert not (policy.episode_dir / "decision_00000").exists()
            preparation = json.loads((policy.episode_dir / "preparation.json").read_text())
            assert preparation["measured_before_episode_clock"]
            assert preparation["startup_ms"] > 0
        assert policy.predict(observation).action.value == 1
        assert json.loads(credentials.read_text())["claudeAiOauth"]["expiresAt"] == 2
        first_process = policy.process
        first_session = policy.session_id
        first_runtime = Path(policy.runtime.name)
        assert policy.predict(observation).action.value == 0
        assert policy.process is first_process
        saved_input = (policy.episode_dir / "decision_00000/input.json").read_text()
        assert "PRIVILEGED_STATE_SENTINEL" not in saved_input
        assert "987654321" not in saved_input
        if realtime:
            assert "The game continues while you think" in saved_input
            assert "OpenVLA" not in saved_input
        np.testing.assert_array_equal(
            Image.open(policy.episode_dir / "decision_00000/observation.png"),
            Image.fromarray(frame).resize((224, 224)),
        )
        policy.reset_state(0)
        assert first_process.poll() == 0
        assert not first_runtime.exists()
        observation.metadata["episode_id"] = 1
        if realtime:
            policy.prepare_episode(1)
        assert policy.predict(observation).action.value == 1
        assert policy.session_id != first_session
        final_process = policy.process
        final_runtime = Path(policy.runtime.name)
    finally:
        policy.close()
    assert final_process.poll() == 0
    assert not final_runtime.exists()


def test_completed_non_json_turn_is_classified_and_keeps_native_result(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config = load_config(Path(__file__).resolve().parents[2] / "configs/experiments/coding_agent/flappy_claude_haiku55_low.yaml")
    config["latency"] = {"method": "zero"}
    config["logging"]["output_dir"] = str(tmp_path / "output")
    policy = ClaudeCodePolicyRunner(config, ActionResolver(config["env"]))
    policy.process = SimpleNamespace(pid=123)
    policy.episode_id = policy.decision_id = 0
    policy.session_id = "test-session"
    policy.episode_dir = tmp_path / "episode"
    policy.episode_dir.mkdir()
    policy.workspace = tmp_path / "workspace"
    (policy.workspace / "observations").mkdir(parents=True)
    monkeypatch.setattr(policy, "_initial_description", lambda: "Task.")
    answer = 'Bird is falling.\n{"action": 1}'
    monkeypatch.setattr(policy, "_request_action", lambda *args: (answer, {"result": answer, "is_error": False}))
    observation = Observation(None, 0, 0, {
        ENV_RAW_RGB_FRAME_STACK_INFO_KEY: np.zeros((1, 8, 8, 3), dtype=np.uint8), "obs_id": 0,
    })
    with pytest.raises(InvalidAgentAction):
        policy.predict(observation)
    record = json.loads((policy.episode_dir / "decision_00000/call.json").read_text())
    assert record["result"]["result"] == answer
    assert record["status"] == "failed"
    assert "action" not in record
    def timeout(*args):
        raise TimeoutError("native timeout")

    monkeypatch.setattr(policy, "_request_action", timeout)
    policy.decision_id = 1
    with pytest.raises(TimeoutError):
        policy.predict(observation)
