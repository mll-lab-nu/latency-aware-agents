"""Long-lived research contracts: ownership, isolation, replay and promotion."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import threading

import numpy as np
from PIL import Image
import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.core.types import Action
from latency_bench.envs.adapters.dummy import DummyEnvAdapter
from latency_bench.executors.factory import build_executor
from latency_bench.harness.candidate import CandidateRuntime
from latency_bench.harness.codex import check_limits
from latency_bench.harness.history import PublicHistory
from latency_bench.policy.base import ConstantPolicyRunner
from latency_bench.utils.io import write_json
from scripts.experiments.asterix_agent import research
from scripts.experiments.asterix_agent.research_contract import (
    apply_proposal, candidate_config, freeze_candidate, promotion_decision, read_candidate,
)


CONFIG = Path(__file__).resolve().parents[2] / "configs/experiments/asterix_agent/reproduce.yaml"


@pytest.fixture
def parent(tmp_path):
    root = tmp_path / "parent"
    root.mkdir()
    (root / "prompt.txt").write_text("Choose a legal action.")
    (root / "strategy.py").write_text(
        'def format_prompt(base, snapshot):\n    return base + str(snapshot["frame"])\n'
        'def should_request(snapshot):\n    return snapshot["frame"] % 8 == 0\n')
    (root / "diagnostic.py").write_text('def verify(snapshot):\n    assert snapshot["images"]\n    return True\n')
    write_json(root / "settings.json", {"image_offsets": [4, 2, 0], "action_history": 8,
                                       "plans_enabled": False, "memory_enabled": False, "tools": {},
                                       "request_function": None, "prompt_function": "strategy:format_prompt"})
    return root


def proposal(files):
    return {"status": "candidate", "diagnosis": {"frame": 0, "observation": "Late reversal",
            "hypothesis": "The prompt omits the current action", "expected_effect": "Use the current action"}, "files": files}


@pytest.mark.parametrize("modification", ["../outside.txt", "worker_instructions.txt", "settings.json", "symlink"])
def test_candidate_cannot_change_protected_paths_or_settings(parent, tmp_path, modification):
    if modification == "symlink":
        (parent / "strategy.py").unlink()
        (parent / "strategy.py").symlink_to(tmp_path / "outside.py")
        with pytest.raises(ValueError, match="path"):
            read_candidate(parent)
    else:
        contents = '{"env": {"reward_scale": 100}}' if modification == "settings.json" else "changed"
        with pytest.raises(ValueError):
            apply_proposal(parent, tmp_path / "candidate", proposal({modification: contents}))
    assert not (tmp_path / "outside.txt").exists()


def test_frozen_candidate_and_atomic_promotion_preserve_evaluated_content(parent, tmp_path):
    working = tmp_path / "working"
    assert apply_proposal(parent, working, proposal({"prompt.txt": "Improved prompt"}))
    frozen = tmp_path / "candidate"
    files = freeze_candidate(working, frozen)
    (working / "prompt.txt").write_text("Later untested edit")
    active = tmp_path / "active.json"
    research.promote(active, None, frozen, files)
    assert Path(json.loads(active.read_text())["candidate_root"]) == frozen
    assert (frozen / "prompt.txt").read_text() == "Improved prompt"
    with pytest.raises(RuntimeError, match="changed during"):
        research.promote(active, None, frozen, files)
    previous = json.loads(active.read_text())
    (frozen / "prompt.txt").write_text("Changed frozen content")
    with pytest.raises(RuntimeError, match="after freezing"):
        research.promote(active, previous, frozen, files)
    assert json.loads(active.read_text()) == previous


@pytest.mark.parametrize("condition,expected", [("improved", True), ("tied", False), ("missing", False),
                                                ("interrupted", False), ("checks_failed", False)])
def test_promotion_uses_only_complete_paired_scores(condition, expected):
    episodes = [{"version": version, "seed": seed, "status": "completed", "score": 100 + (version == "candidate") * 10}
                for version in ("parent", "candidate") for seed in (2, 11, 17)]
    if condition == "tied":
        for item in episodes:
            item["score"] = 100
    elif condition == "missing":
        episodes.pop()
    elif condition == "interrupted":
        episodes[-1]["status"] = "incomplete"
        del episodes[-1]["score"]
    result = promotion_decision(episodes, [2, 11, 17], condition != "checks_failed")
    assert result["promoted"] is expected
    if condition in {"missing", "interrupted", "checks_failed"}:
        assert "means" not in result


def test_platform_limit_mode_has_no_early_percentage_stop():
    response = {"ordinaryUsageAllowed": True, "rateLimits": {"primary": {"usedPercent": 99, "windowDurationMins": 300},
                "secondary": None, "spendControlReached": False, "rateLimitReachedType": None}}
    assert check_limits(response, None)["windows"][0]["used_percent"] == 99
    response["ordinaryUsageAllowed"] = False
    with pytest.raises(RuntimeError):
        check_limits(response, None)


def test_request_skipping_preserves_latency_stream_and_public_request_history(tmp_path):
    config = load_config(CONFIG)
    config["logging"]["output_dir"] = str(tmp_path / "output")
    config["executor"]["inference_pool"] = False
    config["latency"] = {"method": "zero"}
    executor = build_executor(config, policy=ConstantPolicyRunner(Action(2)),
                              env=DummyEnvAdapter(env_fps=7.5, max_steps=17, noop_action=Action(0, is_noop=True)))
    calls = []
    sampler = executor.latency_source

    def sample(context):
        calls.append(context["raw_frame"])
        return sampler.sample(context)

    executor.episode_latency_source_factory = None
    executor.latency_source = deepcopy(sampler)
    executor.latency_source.sample = sample
    executor.harness.request_policy = lambda snapshot, arguments: snapshot["frame"] % 8 == 0
    try:
        executor.run_episode(0, seed=2, max_env_steps=17)
    finally:
        executor.close()
    assert calls == [0, 8, 16]
    events = research.records(tmp_path / "output/public/episode_000/events.jsonl")
    assert [event["frame"] for event in events if event["kind"] == "request_skipped"] == [4, 12]
    assert [event["frame"] for event in events if event["kind"] == "request"] == calls
    assert [image["frame"] for image in json.loads((tmp_path / "output/public/episode_000/request_000004.json").read_text())["images"]] == [0, 8, 16]


@pytest.fixture
def bubblewrap():
    configured = load_config(CONFIG)["policy"]["bubblewrap_executable"]
    if not Path(configured).exists():
        pytest.skip("Research sandbox integration requires the configured Linux server")
    return configured


def test_online_code_cannot_read_hidden_files_write_frozen_code_or_use_network(parent, tmp_path, bubblewrap):
    secret = tmp_path / "private_future.json"
    secret.write_text("FUTURE_SENTINEL")
    (parent / "probe.py").write_text(
        'import pathlib, socket\n'
        'def inspect(snapshot, arguments):\n'
        '    assert not pathlib.Path(arguments["secret"]).exists()\n'
        '    assert not pathlib.Path("/home/ubuntu/.codex/auth.json").exists()\n'
        '    try:\n        pathlib.Path("/candidate/prompt.txt").write_text("changed")\n'
        '    except OSError:\n        pass\n'
        '    else:\n        raise AssertionError("Frozen files are writable")\n'
        '    with socket.socket() as connection:\n'
        '        connection.settimeout(0.1)\n'
        '        assert connection.connect_ex(("1.1.1.1", 443)) != 0\n'
        '    return {"frame": snapshot["frame"], "images": len(snapshot["images"])}\n')
    history = PublicHistory(tmp_path / "public", clock=EnvClock(env_fps=7.5, obs_fps=1.875), image_offsets=[0], action_history=8)
    history.start_episode(0)
    history.observe(0, 0, np.zeros((8, 8, 3), dtype=np.uint8))
    runtime = CandidateRuntime(parent, bubblewrap)
    assert runtime.call("probe:inspect", "tool", history.snapshot(0, 0), {"secret": str(secret)}) == {"frame": 0, "images": 1}
    assert secret.read_text() == "FUTURE_SENTINEL"


def test_offline_cycle_replays_real_ale_and_keeps_parent_on_tied_scores(parent, tmp_path, bubblewrap, monkeypatch):
    pytest.importorskip("ale_py")
    pytest.importorskip("stable_baselines3")
    base = load_config(CONFIG)
    base["executor"]["inference_pool"] = False
    base["latency"] = {"method": "zero"}
    base["logging"]["video"]["enabled"] = False
    real_builder = research.build_executor
    monkeypatch.setattr(research, "build_executor", lambda config: real_builder(config, policy=ConstantPolicyRunner(Action(0, is_noop=True))))
    review = tmp_path / "review"
    review_config = candidate_config(base, parent, review, 2, 40)
    research.run_game(review_config, "parent", 2)
    root = tmp_path / "campaign"
    root.mkdir()
    shutil.copytree(parent, root / "parent")
    active = tmp_path / "active.json"
    settings = {"review_episode": str(review), "review_seed": 2, "seeds": [2, 11, 17], "max_env_steps": 40,
                "active_version_file": str(active), "research_seconds": 10, "total_seconds": 120}
    write_json(root / "run_config.json", settings)
    write_json(root / "protocol.json", base)
    write_json(root / "previous_active.json", None)
    monkeypatch.setattr(research, "propose", lambda *args: proposal({"prompt.txt": "Candidate prompt"}))
    monkeypatch.setattr(research, "preflight", lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(research, "regression_checks", lambda root: None)
    summary = research.execute(root)
    assert summary["status"] == "completed"
    assert len(summary["episodes"]) == 6
    assert summary["promotion"]["reason"] == "mean_not_improved"
    assert not active.exists()
    assert json.loads((root / "replay.json").read_text())["matching_images"] == 10
    # A changed prefix must fail physical replay before proposing another candidate.
    first_image = review / "public/episode_000/images/000000.png"
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(first_image)
    with pytest.raises(ValueError, match="image differs"):
        research.replay_episode(base, review, 2)


@pytest.mark.parametrize("outcome", ["no_candidate", "framework_issue", "quota", "deadline", "interrupted"])
def test_stopped_cycle_records_reason_and_preserves_default(parent, tmp_path, monkeypatch, outcome):
    root = tmp_path / "cycle"
    root.mkdir()
    shutil.copytree(parent, root / "parent")
    active = tmp_path / "active.json"
    previous = {"candidate_root": str(parent)}
    write_json(active, previous)
    settings = {"review_episode": str(tmp_path), "review_seed": 2, "seeds": [2, 11, 17], "max_env_steps": 1200,
                "active_version_file": str(active), "research_seconds": 1,
                "total_seconds": 0.05 if outcome == "deadline" else 10}
    write_json(root / "run_config.json", settings)
    write_json(root / "protocol.json", {})
    write_json(root / "previous_active.json", previous)
    monkeypatch.setattr(research, "regression_checks", lambda root: None)
    monkeypatch.setattr(research, "replay_episode", lambda *args: {"status": "passed"})
    monkeypatch.setattr(research, "review_evidence", lambda *args: ({}, []))

    def researcher(*args):
        if outcome in {"no_candidate", "framework_issue"}:
            response = proposal({})
            response["status"] = outcome
            return response
        if outcome == "quota":
            raise RuntimeError("Ordinary ChatGPT subscription usage is unavailable")
        if outcome == "deadline":
            threading.Event().wait(2)
        raise KeyboardInterrupt()

    monkeypatch.setattr(research, "propose", researcher)
    if outcome in {"no_candidate", "framework_issue"}:
        assert research.execute(root)["status"] == outcome
    else:
        with pytest.raises((RuntimeError, TimeoutError, KeyboardInterrupt)):
            research.execute(root)
    summary = json.loads((root / "summary.json").read_text())
    assert not summary["promotion"]["promoted"]
    assert not summary["episodes"]
    assert json.loads(active.read_text()) == previous
    assert (root / "REPORT.md").exists()
    assert "Completed game episodes: 0/6." in (root / "REPORT.md").read_text()
