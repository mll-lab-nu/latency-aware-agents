"""airhockey's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py after
the generic suite), from the legacy scripts/olympiad/airhockey/test_airhockey.py. Run alone (in the sport's image):
python3 -m core.testkit tests sports/airhockey/plugin.py."""
import os
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from core import local
from core.episode import load_plugin
from core.outcome import decide
from core.sport import FINISHED, MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
import hockey  # noqa: E402  (the plugin's directory is on sys.path now)
from hockey import DT, OTHER, Hockey  # noqa: E402

DUEL = MatchConfig(sport="airhockey", format="duel", seats=("A", "B"), seed=4242, first="A", action_level="low")
SINGLE = MatchConfig(sport="airhockey", format="match", seats=("A",), seed=4711, first="A", action_level="low",
                     level=2)


def place(r, x, y, vx=0., vy=0.):
    r.data.qpos[r.px], r.data.qpos[r.py] = x, y
    r.data.qvel[r.pvx], r.data.qvel[r.pvx + 1] = vx, vy
    mujoco.mj_forward(r.model, r.data)


def play_out(r, policy=None, max_steps=20000):
    """Steps to the end of the game; reviews are handed in at once. policy(r) may set commands each step."""
    while not r.done and r.k < max_steps:
        if r.awaiting_review:
            r.ready()
        if policy:
            policy(r)
        r.step()


def numbers_of(x, out=None):
    out = [] if out is None else out
    if isinstance(x, dict):
        for v in x.values():
            numbers_of(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            numbers_of(v, out)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out.append(float(x))
    return out


def keys_of(x, out=None):
    out = set() if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            out.add(k)
            keys_of(v, out)
    elif isinstance(x, list):
        for v in x:
            keys_of(v, out)
    return out


def test_exploit_bad_and_frozen_actions_are_refused():
    m, layer = SPORT.build(DUEL), SPORT.action_layer("low")
    d0 = m.state_digest()
    for seat, body in (("A", {"x": True, "y": 0}), ("A", {"x": float("nan"), "y": 0}), ("A", {"x": -.5, "y": float("inf")}),
                       ("A", {"x": -.04, "y": 0}), ("A", {"x": -.5, "y": .48}), ("A", {"x": .5, "y": 0}),
                       ("A", {"x": -.95, "y": 0}), ("A", {"x": -.06, "y": .4}), ("A", {"type": "bad", "x": -.5, "y": 0}),
                       ("A", {"type": "target", "x": -.5}), ("A", {"type": "target", "x": -.5, "y": 0, "z": 1}),
                       ("A", {"type": "target", "x": "-.5", "y": 0}), ("A", {"type": "none", "x": -.5}),
                       ("B", {"type": "target", "x": -.5, "y": 0}), ("A", {"type": "target", "x": -.5, "y": 0, "entity": "e"}),
                       ("A", [1, 2])):
        try:
            layer.parse(seat, body, m, {})
            raise AssertionError(f"accepted {seat} {body}")
        except Refused as e:
            assert e.code == 400, (body, e.code)
    assert m.state_digest() == d0 and m.k == 0
    assert layer.parse("A", {"x": -.5, "y": .2}, m, {}) == {"type": "target", "x": -.5, "y": .2}
    assert layer.parse("B", {"type": "target", "x": .5, "y": -.2}, m, {}) == {"type": "target", "x": .5, "y": -.2}
    assert layer.parse("B", {"type": "none"}, m, {}) == {"type": "none"}
    assert 2 * hockey.X_NEAR > 2 * hockey.MALLET_R            # the two mallets can never touch
    m.h.k = round(hockey.CAP_S / DT) - 1                        # frozen at the end: the world refuses to apply
    m.step()
    assert m.phase().state == FINISHED and "rejected" in m.apply("A", {"type": "target", "x": -.5, "y": 0})
    from core.realtime import RealtimeService                   # the core refuses an act while the world is frozen
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    with tempfile.TemporaryDirectory() as tmp:
        svc = RealtimeService(SPORT, PLUGIN, DUEL, Path(tmp))
        try:
            code, out = svc.routes(True)["act"]("A", {"type": "target", "x": -.5, "y": 0})
            assert code == 409 and svc.m.k == 0, (code, out)
        finally:
            for w in svc.workers.values():
                w.close()


def test_exploit_stalling_with_the_puck_is_a_foul_not_a_point():
    r = Hockey(duel=True, first="A")                 # A holds still with the puck in its half
    while not r.results:
        r.step()
    res = r.results[-1]
    assert (res["how"], res["guilty"], res["next"]) == ("seven_second_foul", "A", "B")
    assert abs(res["k"] * DT - hockey.SEVEN_S) < 1e-6
    assert r.scores()["B"]["points"] == 0            # no point for a foul
    r.ready()
    assert r.puck()[0] > 0                            # B has the puck at its spot


def test_exploit_parking_the_puck_out_of_reach_is_a_dead_puck():
    r = Hockey(duel=True)
    place(r, -.04, .42)                              # by the centre line at the side: no mallet reaches it
    assert not (hockey.reachable("A", r.puck()) or hockey.reachable("B", r.puck()))
    while not r.results:
        r.step()
    res = r.results[-1]
    assert (res["how"], res["next"], res["where"]) == ("dead_puck", "faceoff", "out_of_reach")
    assert abs(res["k"] * DT - hockey.DEAD_S) < 1e-6
    r.ready()
    np.testing.assert_allclose(r.puck(), [0., 0.])   # face-off at the centre
    assert hockey.reachable("A", r.puck()) and hockey.reachable("B", r.puck())


def test_exploit_centre_line_stall_is_a_dead_puck():
    r = Hockey(duel=True)
    place(r, 0., 0.)                                 # on the centre line: nobody's puck, no seven-second count
    while not r.results:
        r.step()
    assert (r.results[-1]["how"], r.results[-1]["where"]) == ("dead_puck", "centre_line")
    assert abs(r.results[-1]["k"] * DT - hockey.LINE_S) < 1e-6


def test_exploit_puck_at_the_rails_is_reachable():
    for x in np.linspace(.08, .94, 30):
        for y in np.linspace(-.487, .487, 31):
            for s in "AB":
                assert hockey.reachable(s, [hockey.OWN[s] * x, y]), (s, x, y)


def test_exploit_parking_in_front_of_the_goal_loses():
    def policy(r):
        r.level = 4
        r.set_command("A", r.scripted("A"))
        r.level = 1
        r.set_command("B", [.92, 0.])
    r = Hockey(duel=True, first="B")
    play_out(r, policy)
    assert r.done and r.winner() == "A"
    assert sum(x["guilty"] == "B" for x in r.results) >= 1


def test_exploit_doing_nothing_loses_to_the_scripted_opponent():
    m = SPORT.build(MatchConfig(sport="airhockey", format="match", seats=("A",), seed=1, first="A",
                                action_level="low", level=1))
    while m.phase().state != FINISHED:
        if m.h.awaiting_review:
            m.resume()
        m.step()
    out = decide(m, None, None, ["A"])
    assert out.winner == "opponent" and m.h.winner() == "B", (out, m.scores())


def test_exploit_world_state_hidden():
    for cfg in (DUEL, SINGLE):
        m = SPORT.build(cfg)
        layer = SPORT.action_layer("low")
        for seat, body in (("A", {"x": -.4, "y": .2}), ("B", {"x": .3, "y": -.1})):
            if seat in cfg.seats:
                m.apply(seat, layer.parse(seat, body, m, {}))
        for _ in range(30):
            m.step()
        h = m.h
        puck = h.puck().tolist()
        for seat in cfg.seats:
            other = h.position(OTHER[seat])[:2].tolist()
            obs, spec = m.observe(seat), local.seat_spec(SPORT, cfg, seat, m, plugin=PLUGIN)
            nums_obs, nums_spec = numbers_of(obs), numbers_of(spec)
            for v in puck + other:
                if abs(v) > 1e-3:
                    assert not any(abs(n - v) < 1e-9 or n == round(v, 5) for n in nums_obs), (seat, v)
                    assert not any(abs(n - v) < 1e-9 for n in nums_spec), (seat, v)
            assert not {k for k in keys_of(obs) | keys_of(spec) if "seed" in k.lower()}
            assert cfg.seed not in nums_obs + nums_spec
            bad = {k for k in keys_of(obs) if "puck" in k.lower() or "opp" in k.lower()} - {"next_puck"}
            assert not bad, bad


def test_exploit_mirror_symmetric_robots():
    r = Hockey(duel=True)
    m = r.model
    for f in ("geom_condim", "geom_solref", "geom_solimp", "geom_priority", "geom_friction", "geom_size",
              "geom_contype", "geom_conaffinity", "geom_margin", "geom_solmix"):
        np.testing.assert_array_equal(getattr(m, f)[r.ee["A"]], getattr(m, f)[r.ee["B"]], err_msg=f)
    bodies = {s: [i for i in range(m.nbody) if (m.body(i).name or "").startswith(f"iiwa_{n}/")]
              for s, n in (("A", 1), ("B", 2))}
    np.testing.assert_array_equal(m.body_mass[bodies["A"]], m.body_mass[bodies["B"]])
    np.testing.assert_array_equal(m.body_inertia[bodies["A"]], m.body_inertia[bodies["B"]])
    for f in ("jnt_range", "dof_damping", "dof_frictionloss", "dof_armature"):
        a = getattr(m, f)[r.ids["A"]] if f.startswith("jnt") else getattr(m, f)[r.dadr["A"]]
        b = getattr(m, f)[r.ids["B"]] if f.startswith("jnt") else getattr(m, f)[r.dadr["B"]]
        np.testing.assert_array_equal(a, b, err_msg=f)
    np.testing.assert_array_equal(m.actuator_ctrlrange[r.acts["A"]], m.actuator_ctrlrange[r.acts["B"]])
    np.testing.assert_allclose(r.position("A")[:2], -r.position("B")[:2], atol=1e-9)
    paths = {}
    for s in ("A", "B"):
        g = Hockey(duel=True, seed=5)
        sg = hockey.OWN[s]
        place(g, sg * .58, sg * .07)
        g.set_command(s, [sg * .40, sg * .05])
        g.set_command(OTHER[s], [-sg * .82, 0.])
        p = []
        for _ in range(150):
            g.step()
            p.append(g.puck() * (1 if s == "A" else -1))
        paths[s] = np.array(p)
    assert np.abs(paths["A"][-1] - paths["A"][0]).max() > .3      # the puck was hit
    np.testing.assert_allclose(paths["A"], paths["B"], atol=1e-6)
    ca, cb = (SPORT.build(DUEL).cameras(s)[0] for s in "AB")     # the cameras: half a turn about the table centre
    for k in ("pos", "lookat", "up"):
        np.testing.assert_allclose(np.multiply(ca[k], [-1, -1, 1]), cb[k], atol=1e-12)


def test_exploit_no_draw_tiebreak_chain():
    """A finished game always has a winner: more goals, else the last goal, else territory, else the player who did
    not have the puck first; the plugin's chain gives Hockey.winner() in every case."""
    m = SPORT.build(MatchConfig(sport="airhockey", format="duel", seats=("A", "B"), seed=1, first="A",
                                action_level="low"))
    r = m.h
    r.k = round(hockey.CAP_S / DT) - 1
    m.step()
    assert r.done and r.results[-1]["how"] == "time_cap" and m.phase().state == FINISHED

    def winner():
        out = decide(m, None, None, ["A", "B"])
        assert out.winner == r.winner(), (out, r.winner())
        return out.winner, out.decided_by
    r.half_steps = {"A": 5, "B": 5}
    assert winner() == ("B", "first_possession")                     # A had the puck first
    r.half_steps = {"A": 7, "B": 5}
    assert winner() == ("B", "territory")                            # the puck was more in A's half
    r.results = [dict(rally=1, how="goal", winner="A", guilty=None, next="B", k=1, steps=1),
                 dict(rally=2, how="goal", winner="B", guilty=None, next="A", k=2, steps=1)]
    assert winner() == ("B", "last_goal")                            # 1:1, B scored last
    r.results.append(dict(rally=3, how="goal", winner="A", guilty=None, next="B", k=3, steps=1))
    assert winner() == ("A", "goals")                                # 2:1
    for first in ("A", "B"):                                         # a whole game without any play
        g = Hockey(duel=True, first=first)
        play_out(g)
        assert g.done and g.winner() in ("A", "B") and len(g.results) <= hockey.MAX_RALLIES
    g = Hockey(duel=True)                                            # seven goals end the game at once
    for _ in range(hockey.POINTS_TO_WIN):
        assert not g.done
        if g.awaiting_review:
            g.ready()
        place(g, 1.0, 0., 1.)
        g.step()
    assert g.done and g.decided_by == "score" and g.winner() == "A" and g.scores()["A"]["points"] == 7
