"""Long-lived native, observation and scheduler contracts for the three frozen sports.

Each sport runs in a fresh interpreter because upstream imports share module names.
This suite owns the adapter boundary, not the upstream physics or opponent quality.
"""

import subprocess
import sys
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco", reason="requires the optional RoboOlympics simulator runtime")
if mujoco.__version__ != "3.3.7":
    pytest.skip("requires pinned MuJoCo 3.3.7", allow_module_level=True)


@pytest.mark.parametrize("sport", ["badminton", "soccer", "soccer_ssl"])
def test_frozen_match_contracts(sport, tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), sport, str(tmp_path)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_contracts(sport, output_dir):
    import numpy as np

    from latency_bench.core.config import load_config
    from latency_bench.core.types import Action
    from latency_bench.envs.adapters.roboolympics import RoboOlympicsAdapter, load_sport
    from latency_bench.executors.factory import build_executor
    from latency_bench.logging.action_trace_replay import load_step_records_by_episode
    from core.sport import FINISHED, UNIT_OVER, parse_action

    root = Path(__file__).resolve().parents[3]
    wait = Action(value=None, name="wait", is_noop=True)
    config = load_config(root / f"configs/examples/roboolympics/{sport}.yaml")
    commands = {
        "badminton": {"type": "move", "base": [0.12, -1.6]},
        "soccer": {"type": "walk", "entity": "1", "v": [0.1, 0, 0]},
        "soccer_ssl": {"type": "dribble", "entity": "1", "rpm": 5000},
    }
    command = commands[sport]
    env = RoboOlympicsAdapter(
        sport=sport, level=config["env"]["level"], seed=config["experiment"]["seed"],
        max_env_steps=10000, noop_action=wait,
    )
    try:
        first = env.reset()
        assert first.metadata["spec"]["timing"]["reaction_delay_steps"] == 0
        if sport == "badminton":
            assert first.metadata["spec"]["act"]["lead_steps"] == 0
        frame = env.render_game_frame()
        assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
        assert set(env.match.observe("A")) <= env._allowed
        assert "truth" not in first.data["state"]
        assert env.match.dt * 1000 == env.frame_ms

        direct = env.sport.build(env._cfg)
        layer = env.sport.action_layer(env._cfg.action_level)
        if sport == "badminton":
            layer.lead = 0
        for step in range(12):
            before = direct.scores()
            if step == 0:
                parsed = parse_action(layer, "A", command, direct, {})
                job = layer.snapshot("A", parsed, direct, step)
                prepared = layer.prepare(job) if job is not None else None
                low, _ = layer.lower("A", parsed, prepared, direct)
                assert "rejected" not in direct.apply("A", low)
            direct.step()
            result = env.step(Action(command) if step == 0 else wait)
            after = direct.scores()
            assert result.reward == after["A"] - before["A"] - after["opponent"] + before["opponent"]
            assert env.env_step == direct.k
            assert env.match.state_digest() == direct.state_digest(), (sport, step)
            if step == 0:
                assert not result.info.get("invalid_action"), result.info
                if sport == "badminton":
                    assert result.info["action_result"]["effect_step"] == 0
                    assert env.match.deferred["A"] is None
        expected_digest = env.match.state_digest()
        env.reset()
        for step in range(12):
            env.step(Action(command) if step == 0 else wait)
        assert env.match.state_digest() == expected_digest

        result = env.step(Action({"type": "unknown"}))
        assert result.info["invalid_action"] and result.info["action_error"]
        assert env.env_step == 13
        original_observe = env.match.observe
        env.match.observe = lambda seat: {**original_observe(seat), "hidden_test_state": 1}
        with pytest.raises(ValueError, match="undeclared fields"):
            env.observe()
        env.match.observe = original_observe

        if sport == "badminton":
            for _ in range(1000):
                result = env.step(wait)
                if result.info.get("soft_reset"):
                    break
            assert result.info["soft_reset"]
        elif sport == "soccer":
            native_step = env.match.step

            def stop_at_boundary():
                events = native_step()
                env.match.w._stop("half_time", "kickoff")
                return events

            env.match.step = stop_at_boundary
            result = env.step(wait)
            env.match.step = native_step
            assert result.info["soft_reset"]
        else:
            env.match.w.referee._end_half()
            result = env.step(wait)
            assert result.info["soft_reset"]
        assert not result.done and env.match.phase().state != UNIT_OVER

        # A tied regulation score can still produce a native loss.
        env.reset()
        if sport == "badminton":
            env.match.g.done = True  # tied score, no points: first receiver wins
        elif sport == "soccer":
            native_step = env.match.step

            def finish_at_boundary():
                events = native_step()
                env.match.w._finish("match_over", "pso", winner="B")
                return events

            env.match.step = finish_at_boundary
        else:
            ref = env.match.w.referee
            ref.shootout = {"goals": {"blue": 0, "yellow": 1}, "coin": None}
            ref._end_match("SHOOTOUT")
        result = env.step(wait)
        assert result.done and not result.truncated and env.match.phase().state == FINISHED
        assert result.info["winner"] == "opponent"
        assert result.info["task_metrics"]["success"] is False
        assert result.reward == 0

        renderer = env._renderer
        env.close()
        env.close()
        if sport == "badminton":
            assert renderer.r._gl_context is None
        else:
            assert renderer.renderer._r == {}
        with pytest.raises(ValueError, match="separate processes"):
            load_sport("tabletennis")
    finally:
        env.close()

    # Actual executor traces protect zero/100 ms release timing and one-shot commands.
    for latency in (0, 100):
        cfg = load_config(root / f"configs/examples/roboolympics/{sport}.yaml")
        cfg["policy"] = {"type": "constant", "action": command}
        cfg["latency"] = {"method": "fixed", "fixed_latency_ms": latency}
        delay_steps = round(latency * cfg["env"]["env_fps"] / 1000)
        stride = round(cfg["env"]["env_fps"] / cfg["env"]["obs_fps"])
        limit = delay_steps + stride + 1
        cfg["evaluation"]["max_env_steps"] = limit
        cfg["logging"]["output_dir"] = str(output_dir / str(latency))
        executor = build_executor(cfg)
        try:
            metrics = executor.run_episode(0, seed=cfg["experiment"]["seed"], max_env_steps=limit)
            records = load_step_records_by_episode(cfg["logging"]["output_dir"])[0]
            applied = [r for r in records if not r.active_action.is_noop]
            assert [r.env_step for r in applied] == [delay_steps, delay_steps + stride]
            assert all(not r.info.get("invalid_action") for r in applied)
            assert metrics.mean_latency_ms == latency
            assert metrics.metadata["task_metrics"]["truncated"] is True
            assert metrics.metadata["task_metrics"]["match_finished"] is False
        finally:
            executor.close()


if __name__ == "__main__":
    _check_contracts(sys.argv[1], Path(sys.argv[2]))
