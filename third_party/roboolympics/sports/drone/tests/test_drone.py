"""drone's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py after the
generic suite), from the legacy scripts/olympiad/drone/test_drone.py. Run alone (in the sport's image):
python3 -m core.testkit tests sports/drone/plugin.py."""
import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from core.episode import load_plugin
from core.sport import MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
import race  # noqa: E402  (the plugin's directory is on sys.path now)
from race import BOUNDS, DT, Event, Heat, Race  # noqa: E402

DUEL = MatchConfig(sport="drone", format="duel", seats=("A", "B"), seed=1, first="A", action_level="low")
SINGLE = MatchConfig(sport="drone", format="racing", seats=("A",), seed=123457, first="A",
                     action_level="low", level=3)
FORWARD = {"velocity": [2.0, 0.0, 0.0], "yaw_rate": 0.0, "duration": 1.0}
DOWN = {"type": "fly", "vx": 0.0, "vy": 0.0, "vz": -2.0, "yaw_rate": 0.0, "duration": 1.0}


def fly(r, action, ticks):
    for k in range(ticks):
        if r.done:
            break
        if k % 50 == 0:
            r.act(action)
        r.step()


def test_exploit_fly_over_or_under_panel():
    assert race.PANEL_BOTTOM <= 0.0 and race.PANEL_BOTTOM + race.PANEL_HEIGHT >= BOUNDS["z"][1]
    panel = Race(303, 3).panels[0]
    for z in (BOUNDS["z"][1] - 0.15, BOUNDS["z"][0] + 0.06):     # just under the ceiling / just above the floor
        r = Race(303, 3)
        r.data.qpos[:3] = [panel.center[0] - 0.6, panel.center[1], z]
        r.data.qvel[:] = 0
        mujoco.mj_forward(r.model, r.data)
        fly(r, FORWARD, 200)
        assert r.reason in ("collision", "out_of_bounds"), (z, r.reason)
        assert r.data.qpos[0] < panel.center[0] + 0.1


def test_exploit_skip_or_reverse_gate_does_not_count():
    r = Race(101, 1)
    g = r.gates[0]
    n = np.array([math.cos(g.yaw), math.sin(g.yaw), 0.0])
    assert g.crossed(g.center - 0.1 * n, g.center + 0.1 * n) is not None
    assert g.crossed(g.center + 0.1 * n, g.center - 0.1 * n) is None             # backwards
    lateral = np.array([-n[1], n[0], 0.0]) * (g.width / 2 - race.RADIUS + 0.02)
    assert g.crossed(g.center + lateral - 0.1 * n, g.center + lateral + 0.1 * n) is None   # inside the margin
    g2 = r.gates[1]                                     # passing gate 2 first does not count
    n2 = np.array([math.cos(g2.yaw), math.sin(g2.yaw), 0.0])
    r.data.qpos[:3] = g2.center - 0.5 * n2
    mujoco.mj_forward(r.model, r.data)
    fly(r, {"velocity": [float(2.0 * n2[0]), float(2.0 * n2[1]), 0.0], "yaw_rate": 0.0, "duration": 1.0}, 60)
    assert r.passed == 0


def test_exploit_stalling_and_leaving_lose():
    r = Race(101, 1)                                    # hovering: the time limit, no result
    fly(r, {"velocity": [0.0, 0.0, 0.0], "yaw_rate": 0.0, "duration": 1.0}, round(r.cfg["limit"] / DT) + 5)
    assert r.reason == "time_limit" and r.result()["finished"] is False and r.passed == 0
    r = Race(101, 1)                                    # climbing out of the arena
    fly(r, {"velocity": [0.0, 0.0, 2.0], "yaw_rate": 0.0, "duration": 1.0}, 600)
    assert r.reason == "out_of_bounds"
    r = Race(101, 1)                                    # diving: the floor's safety line (z > 0.15 m)
    fly(r, {"velocity": [0.0, 0.0, -2.0], "yaw_rate": 0.0, "duration": 1.0}, 300)
    assert r.reason in ("collision", "out_of_bounds") and r.passed == 0


def test_exploit_review_stalling_forfeits():
    """Both seats dive (a tied heat 1, review due); A reviews, B does not: B loses by ready_timeout (valid)."""
    from core.realtime import RealtimeService
    env = {"READY_TIMEOUT_S": "1.0", "START_TIMEOUT_S": "30", "BARRIER_LEAD_S": "0.05", "RCB_ACT_RULE": "program",
           "PLAYER_MODE_A": "oracle", "PLAYER_MODE_B": "oracle", "RCB_BROADCAST": "0"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        cfg = MatchConfig(sport="drone", format="duel", seats=("A", "B"), seed=1, first="A",
                          action_level="low", slowdown=0.05)
        with tempfile.TemporaryDirectory() as tmp:
            svc = RealtimeService(SPORT, PLUGIN, cfg, Path(tmp))
            threading.Thread(target=svc.loop, daemon=True).start()
            post = svc.routes(True)
            th = [threading.Thread(target=post["start"], args=(s, {}), daemon=True) for s in "AB"]
            for t in th:
                t.start()
            for t in th:
                t.join(30)
            for s in "AB":
                assert post["act"](s, DOWN)[0] == 200
            t0 = time.time()
            while svc.phase != "review" and time.time() - t0 < 120:
                time.sleep(0.05)
            assert svc.phase == "review" and svc.m.ev.closed[0]["winner"] is None, svc.phase
            assert post["act"]("A", DOWN)[0] == 409                      # frozen during the review
            threading.Thread(target=post["ready"], args=("A", {"why": "w", "plan": "p"}), daemon=True).start()
            t0 = time.time()
            while not svc.finalized and time.time() - t0 < 120:
                time.sleep(0.1)
            d = json.loads((Path(tmp) / "final.json").read_text())
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert d["reason"] == "ready_timeout" and d["forfeiting"] == "B" and d["match_winner"] == "A", d["reason"]
    assert d["valid_trial"] and d["deterministic"], d["errors"]


def test_exploit_bad_and_frozen_actions_are_refused():
    m, layer = SPORT.build(DUEL), SPORT.action_layer("low")
    d0 = m.state_digest()
    good = {"type": "fly", "vx": 1.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0, "duration": 0.5}
    for body in ({**good, "vx": 3.5, "vy": 3.5}, {**good, "vz": 2.5}, {**good, "duration": 2.0},
                 {**good, "duration": 0.01}, {**good, "yaw_rate": float("nan")}, {**good, "vx": "1"},
                 {k: v for k, v in good.items() if k != "vz"}, {**good, "spin": 1.0}, {"type": "goto", "x": 3.0},
                 {"velocity": [1, 0, 0], "yaw_rate": 0, "duration": 0.5}):
        try:
            layer.parse("A", body, m, {})
            raise AssertionError(f"accepted {body}")
        except Refused as e:
            assert e.code == 400, (body, e.code)
    assert m.state_digest() == d0 and m.k == 0
    assert layer.parse("A", good, m, {})["vx"] == 1.0
    m.apply("A", DOWN)                                  # A dives: its flight ends, B flies on
    while not m.ev.heat.races["A"].done:
        m.step()
    try:
        layer.parse("A", good, m, {})
        raise AssertionError("an act after the own flight ended was accepted")
    except Refused as e:
        assert e.code == 409
    assert "rejected" in m.apply("A", good)             # and the world ignores it (as the legacy service)


def _numbers(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers(v, out)
    elif isinstance(obj, float):
        out.append(obj)
    return out


def test_exploit_world_state_hidden():
    from core import local
    for cfg in (SINGLE, MatchConfig(sport="drone", format="duel", seats=("A", "B"), seed=123457, first="A",
                                    action_level="low")):
        m = SPORT.build(cfg)
        for _ in range(30):
            m.step()
        spec = local.seat_spec(SPORT, cfg, "A", m, plugin=PLUGIN)
        obs = m.observe("A")
        text = json.dumps([spec, obs], default=float)
        assert '"seed"' not in text and str(cfg.seed) not in text
        r = m.ev.heat.races["A"]
        hidden = [float(x) for g in r.gates for x in g.center[:2]]
        hidden += [float(x) for p in r.panels for x in p.center[:2]]
        hidden += [float(x) for h in r.shutters for x in (h.period, h.phase)]
        hidden += [float(x) for x in r.wind_phase]
        shown = _numbers([spec, obs], [])
        for h in hidden:
            assert not any(abs(h - v) < 1e-9 for v in shown), h
    assert "seed" not in Race(123457, 2).spec()["configuration"]


def test_exploit_separate_identical_worlds():
    h = Heat(1, 2, 1, ("A", "B"))
    a, b = h.races["A"], h.races["B"]
    assert a.xml == b.xml and a.fingerprint == b.fingerprint and a.model is not b.model
    assert np.array_equal(a.wind_phase, b.wind_phase)
    assert [(s.period, s.phase) for s in a.shutters] == [(s.period, s.phase) for s in b.shutters]
    for k in range(300):                               # the same actions give the same flight on either seat
        if k % 40 == 0:
            act = {"velocity": [1.5, 0.3 * math.sin(k), 0.1], "yaw_rate": 0.2, "duration": 0.5}
            h.act("A", act)
            h.act("B", act)
        h.step()
        if h.done:
            break
    assert a.state_hash() == b.state_hash() and h.checkpoints["A"] == h.checkpoints["B"]
    m1, m2 = SPORT.build(DUEL), SPORT.build(DUEL)       # B's commands never reach A's world or observation
    m2.apply("B", {"type": "fly", "vx": 2.0, "vy": 1.0, "vz": 0.5, "yaw_rate": 0.5, "duration": 1.0})
    for _ in range(150):
        m1.step()
        m2.step()
    assert m1.ev.heat.races["A"].state_hash() == m2.ev.heat.races["A"].state_hash()
    assert m1.observe("A") == m2.observe("A") and m1.cameras("A") == m2.cameras("A")
    assert m1.render_state() == m2.render_state()       # the course scene holds no drone at all
    assert m1.ev.heat.races["B"].state_hash() != m2.ev.heat.races["B"].state_hash()


def _closed(winners, ra=None, rb=None):
    def res(finished=False, t=None, gates=0, last=None):
        return dict(finished=finished, finish_time_s=t if finished else None, gates_passed=gates,
                    last_gate_time_s=last if gates else None)
    return [dict(number=i + 1, winner=w, results={"A": ra or res(), "B": rb or res()}) for i, w in enumerate(winners)]


def test_exploit_no_draw_tiebreak_chain():
    res = lambda g, last: dict(finished=False, finish_time_s=None, gates_passed=g, last_gate_time_s=last)  # noqa: E731
    m = SPORT.build(DUEL)
    names = [n for n, _ in m.tiebreaks()]
    assert names == ["heat_wins", "most_heat_wins", "best_results", "last_heat_winner", "lot"]
    ev = m.ev
    ev.over = True
    for closed, want in ((_closed(["A", None, "A"]), ("A", "heat_wins")),
                         (_closed(["A", None, None, None, None]), ("A", "most_heat_wins")),
                         (_closed([None] * 5, res(2, 9.0), res(3, 12.0)), ("B", "best_results"))):
        ev.closed = closed
        rule = dict(m.tiebreaks())[want[1]]
        assert ev.duel_winner() == want and rule(m) == want[0]
    for seed in range(20):                              # nothing separates them: the seeded lot still decides
        e2 = Event(seed, 2, ("A", "B"))
        e2.over, e2.closed = True, _closed([None] * 5)
        assert e2.duel_winner()[0] in ("A", "B") and e2.duel_winner()[1] == "lot"
    ev2 = Event(2, 2, ("A", "B"))                       # a tied heat is re-flown
    for _ in range(3):
        h = ev2.heat
        for r in h.races.values():
            r.reason, r.done = "x", True
        h.done, h.end = True, "all_done"
        ev2.close_heat()
        assert ev2.closed[-1]["winner"] is None
    assert not ev2.over and ev2.heat.number == 4
