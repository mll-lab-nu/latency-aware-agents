"""Long-lived table-tennis adapter contracts, exercised against the optional upstream simulator."""

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="requires the optional RoboOlympics simulator runtime")
if mujoco.__version__ != "3.3.7":
    pytest.skip("requires the pinned RoboOlympics MuJoCo 3.3.7 runtime", allow_module_level=True)

from latency_bench.core.config import load_config
from latency_bench.core.types import Action
from latency_bench.envs.adapters.roboolympics import SPORT, RoboOlympicsTableTennisAdapter
from latency_bench.executors.factory import build_executor
from latency_bench.logging.action_trace_replay import load_step_records_by_episode

from core.sport import FINISHED, UNIT_OVER, parse_action


WAIT = Action(value=None, name="wait", is_noop=True)
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def env():
    adapter = RoboOlympicsTableTennisAdapter(level=1, seed=7, max_env_steps=10000, noop_action=WAIT)
    try:
        adapter.reset(seed=7)
        yield adapter
    finally:
        adapter.close()


def test_native_trace_and_seeded_reset_match_upstream(env):
    direct = SPORT.build(env._cfg)
    layer = SPORT.action_layer("high")
    actions = {0: {"type": "move", "base": [0.1, -1.8]}, 25: {"type": "none"}}
    for index in range(60):
        command = actions.get(index)
        if command is not None:
            parsed = parse_action(layer, "A", command, direct, {})
            low, _ = layer.lower("A", parsed, None, direct)
            direct.apply("A", low)
        direct.step()
        result = env.step(WAIT if command is None else Action(command))
        assert result.reward == 0
        assert env.env_step == direct.k
        assert env.match.scores() == direct.scores()
        np.testing.assert_array_equal(env.match.R.s.d.qpos, direct.R.s.d.qpos)
    final_qpos = env.match.R.s.d.qpos.copy()
    env.reset(seed=7)
    for index in range(60):
        command = actions.get(index)
        env.step(WAIT if command is None else Action(command))
    np.testing.assert_array_equal(env.match.R.s.d.qpos, final_qpos)


def test_wait_preserves_command_and_none_cancels_it(env):
    env.match.apply("A", {"type": "hold"})
    env.step(WAIT)
    assert env.match.R.cmd is None
    env.step(Action({"type": "none"}))
    assert env.match.R.cmd["type"] == "slew"
    result = env.step(Action({"type": "unknown"}))
    assert result.info["invalid_action"] is True
    assert result.info["action_error"]
    assert env.env_step == 3


def test_observation_is_public_rgb_and_close_releases_renderer(env, monkeypatch):
    observation = env.observe()
    assert observation.env_step == 0
    assert observation.data["state"]["step"] == 0
    assert "q" in observation.data["state"]
    assert "ball_pos" not in observation.data["state"]
    frame = env.render_game_frame()
    assert frame.shape == (720, 1920, 3)
    assert frame.dtype == np.uint8
    assert env.observe() is observation
    original_observe = env.match.observe
    monkeypatch.setattr(env.match, "observe", lambda seat: {**original_observe(seat), "hidden_ball_position": [0, 0, 0]})
    env.step(WAIT)
    with pytest.raises(ValueError, match="undeclared fields"):
        env.observe()
    renderer = env._renderer.r
    env.close()
    env.close()
    assert renderer._gl_context is None


def test_points_resume_and_match_termination_preserve_score_reward(env):
    total_reward = 0
    boundaries = 0
    for _ in range(10000):
        result = env.step(WAIT)
        total_reward += result.reward
        if result.info.get("soft_reset"):
            boundaries += 1
            assert not result.done
            assert env.match.phase().state != UNIT_OVER
        if result.done or result.truncated:
            break
    assert boundaries > 0
    assert result.done and not result.truncated
    assert env.match.phase().state == FINISHED
    scores = env.match.scores()
    assert total_reward == scores["A"] - scores["opponent"]
    assert result.info["task_metrics"]["match_finished"] is True


def test_disqualification_uses_native_winner_despite_score_lead(env):
    env.match.R.score[:] = [4, 0]
    env.match.R.disqualified = 0
    for _ in range(1000):
        result = env.step(WAIT)
        if result.done:
            break
    assert result.done
    assert env.match.scores()["A"] > env.match.scores()["opponent"]
    assert result.info["winner"] == "opponent"
    assert result.info["task_metrics"]["success"] is False


def test_fixed_latency_applies_once_and_limit_is_truncated(tmp_path):
    config = load_config(ROOT / "configs/examples/roboolympics/tabletennis_100ms.yaml")
    config["policy"] = {"type": "constant", "action": {"type": "move", "base": [0.1, -1.8]}}
    config["evaluation"]["max_env_steps"] = 16
    config["logging"]["output_dir"] = str(tmp_path)
    executor = build_executor(config)
    try:
        metrics = executor.run_episode(0, seed=7, max_env_steps=16)
        records = load_step_records_by_episode(tmp_path)[0]
        assert all(record.active_action.is_noop for record in records[:10])
        assert [record.env_step for record in records if not record.active_action.is_noop] == [10, 15]
        assert metrics.mean_latency_ms == 100
        assert metrics.metadata["task_metrics"]["truncated"] is True
        assert metrics.metadata["task_metrics"]["match_finished"] is False
    finally:
        executor.close()


def test_delayed_commands_do_not_cross_point_boundaries(tmp_path):
    config = load_config(ROOT / "configs/examples/roboolympics/tabletennis_100ms.yaml")
    config["policy"] = {"type": "constant", "action": {"type": "move", "base": [0.1, -1.8]}}
    config["logging"]["output_dir"] = str(tmp_path)
    executor = build_executor(config)
    try:
        executor.run_episode(0, seed=7, max_env_steps=3000)
        last_boundary = -1
        boundaries = 0
        for record in load_step_records_by_episode(tmp_path)[0]:
            source = record.info["active_action_obs_env_step"]
            if not record.active_action.is_noop:
                assert source > last_boundary
            if record.info.get("soft_reset"):
                last_boundary = record.env_step
                boundaries += 1
        assert boundaries > 0
    finally:
        executor.close()
