"""Long-lived native worker lifecycle and subscription boundary contracts."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.harness.codex import HarnessCodexRunner, check_limits
from latency_bench.harness.history import PublicHistory


def test_worker_reuses_process_starts_fresh_threads_and_preserves_shared_home(tmp_path, monkeypatch):
    config = load_config(Path(__file__).resolve().parents[2] / "configs/experiments/asterix_agent/reproduce.yaml")
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.json").write_text("AUTH_SENTINEL")
    (home / "existing-state").write_text("SHARED_SENTINEL")
    instructions = tmp_path / "instructions.txt"
    instructions.write_text("GAME_WORKER_INSTRUCTIONS")
    (tmp_path / "bin").mkdir()
    version = tmp_path / "bin/codex"
    version.write_text("#!/bin/sh\necho codex-cli-fixture\n")
    version.chmod(0o755)
    config["logging"]["output_dir"] = str(tmp_path / "output")
    config["policy"].update(codex_home=str(home), workspace_root=str(tmp_path / "workspaces"),
                            runtime_root=str(tmp_path), model_instructions_file=str(instructions), package_root=str(tmp_path))
    fixture = Path(__file__).with_name("fake_codex.py")

    def launch(self, workspace, runtime):
        self._archive_home(self.episode_dir / "prior_native_state")
        return [sys.executable, str(fixture), str(workspace)]

    monkeypatch.setattr(HarnessCodexRunner, "_sandbox_command", launch)
    history = PublicHistory(tmp_path / "public", clock=EnvClock(env_fps=7.5, obs_fps=1.875),
                            image_offsets=[4, 2, 0], action_history=8)
    history.start_episode(0)
    for frame in range(0, 17, 4):
        history.observe(frame, frame, np.zeros((16, 16, 3), dtype=np.uint8))
    observation = history.request(16, 16)
    policy = HarnessCodexRunner(config, ActionResolver(config["env"]))
    try:
        assert policy.predict(observation).action.value == 1
        process = policy.process
        first_thread = policy.thread_id
        assert policy.predict(observation).action.value == 2
        assert policy.process is process
        assert policy.thread_id != first_thread
        calls = sorted(policy.episode_dir.glob("decision_*/call.json"))
        assert json.loads(calls[-1].read_text())["result"]["tokenUsage"]["last"]["inputTokens"] == 120
    finally:
        policy.close()
    assert process.poll() == 0
    assert (home / "auth.json").read_text() == "AUTH_SENTINEL"
    assert (home / "existing-state").read_text() == "SHARED_SENTINEL"


@pytest.mark.parametrize("condition", ["stop_line", "no_windows", "ordinary_disabled", "spend_reached", "invalid_usage"])
def test_subscription_guard_stops_before_model_call(condition):
    limits = {"ordinaryUsageAllowed": True, "rateLimits": {
        "primary": {"usedPercent": 20, "windowDurationMins": 10080}, "secondary": None,
        "spendControlReached": False, "rateLimitReachedType": None}}
    assert check_limits(deepcopy(limits), 37)["windows"][0]["used_percent"] == 20
    if condition == "stop_line":
        limits["rateLimits"]["primary"]["usedPercent"] = 37
    elif condition == "no_windows":
        limits["rateLimits"]["primary"] = None
    elif condition == "ordinary_disabled":
        limits["ordinaryUsageAllowed"] = False
    elif condition == "spend_reached":
        limits["rateLimits"]["spendControlReached"] = True
    else:
        limits["rateLimits"]["primary"]["usedPercent"] = float("nan")
    with pytest.raises((ValueError, RuntimeError)):
        check_limits(limits, 37)
