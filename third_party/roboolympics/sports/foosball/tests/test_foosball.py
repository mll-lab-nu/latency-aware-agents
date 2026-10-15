"""foosball's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py after
the generic suite), from the legacy scripts/olympiad/foosball/test_foosball.py. Run alone (in the sport's image):
python3 -m core.testkit tests sports/foosball/plugin.py."""

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

HERE = Path(__file__).resolve().parents[1]
PLUGIN = str(HERE / "plugin.py")
SPORT = load_plugin(PLUGIN)
import foos  # noqa: E402  (the plugin's directory is on sys.path now)
from foos import DT, OTHER, OWN, RODS, Foosball, validate_action  # noqa: E402
from foos_policy import Policy  # noqa: E402

DUEL = MatchConfig(sport="foosball", format="duel", seats=("A", "B"), seed=2, first="A", action_level="low")
MATCH = MatchConfig(sport="foosball", format="match", seats=("A",), seed=123457, first="A", action_level="low", level=2)


def place(r, x, y, vx=0.0, vy=0.0, z=foos.BALL_R):
    r.data.qpos[r.bq : r.bq + 3] = x, y, z
    r.data.qpos[r.bq + 3 : r.bq + 7] = 1, 0, 0, 0
    r.data.qvel[r.bv : r.bv + 6] = vx, vy, 0, 0, 0, 0
    mujoco.mj_forward(r.model, r.data)


def open_play(r):
    r.opening["done"] = True


def policies(r, kw):
    out = {}
    for s in "AB":
        p = Policy(**kw[s])
        mine = r.restart_where[1] == s
        p.reset(("goalie" if r.restart_where[0] == "goalie" else "midfield") if mine else None)
        out[s] = p
    return out


def play_out(r, kw=None, max_steps=40000, delay=4):
    """Steps to the end of the game; reviews are handed in at once. kw: {seat: Policy kwargs} for the seats played by
    the policy on the true ball (a reaction delay ago); seats not in kw do nothing (the legacy test helper)."""
    kw = kw or {}
    pol = policies(r, {s: kw.get(s, {}) for s in "AB"})
    hist = []
    while not r.done and r.k < max_steps:
        if r.awaiting_review:
            r.ready()
            pol = policies(r, {s: kw.get(s, {}) for s in "AB"})
            hist = []
        hist.append(r.ball()[:2].copy())
        hist = hist[-(delay + 3) :]
        if r.k % 2 == 0 and kw:
            P = hist[0]
            W = (hist[2] - hist[0]) / (2 * DT) if len(hist) > 2 else np.zeros(2)
            P = P + W * delay * DT
            for s in kw:
                sg = OWN[s]
                rods = {
                    x: dict(
                        slide=float(r.data.qpos[r.sq[r.rod_index(s, x)]]),
                        angle=float(r.data.qpos[r.aq[r.rod_index(s, x)]]),
                    )
                    for x in RODS
                }
                opp = {x: [sg * y for y in r.figure_y(OTHER[s], x)] for x in RODS}
                c = pol[s].update(
                    r.point_k * DT,
                    (float(sg * P[0]), float(sg * P[1])),
                    (float(sg * W[0]), float(sg * W[1])),
                    rods,
                    opp,
                    opening_done=r.opening["done"] or r.opening["team"] != s,
                )
                r.schedule(
                    s, validate_action(dict(rods={x: {k: float(v) for k, v in d.items()} for x, d in c.items()}))
                )
        r.step()


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


def test_reference_policy_is_the_opponents():
    """builtin_players.py carries a verbatim copy of foos_policy.py (the scripted team's policy) from ROD_X on."""
    src = (HERE / "foos_policy.py").read_text()
    body = src[src.index("ROD_X = ") :]
    bp = (HERE / "builtin_players.py").read_text()
    assert bp.endswith(body), "builtin_players.py's policy differs from foos_policy.py"


def test_exploit_holding_the_ball_is_a_foul_not_a_point():
    """Stalling with the ball on a rod ends within the possession limit; the opponent gets the ball, no point."""
    r = Foosball(duel=True, first="A")
    open_play(r)  # A holds the kick-off ball still on its 5-rod
    while not r.results:
        r.step()
    res = r.results[-1]
    assert (res["how"], res["guilty"], res["group"], res["next"]) == (
        "possession_time",
        "A",
        "midfield",
        ["midfield", "B"],
    )
    assert abs(res["k"] * DT - 10.01) < 1e-6 and r.scores()["B"]["points"] == 0
    for x, group, limit, award in ((-0.225 + 0.45, "attack", 15.0, "goalie"), (-0.45, "goalie_rods", 15.0, "midfield")):
        r = Foosball(duel=True)
        open_play(r)
        place(r, x, 0.05)
        while not r.results:
            r.step()
        res = r.results[-1]
        assert (res["how"], res["guilty"], res["group"], res["next"]) == (
            "possession_time",
            "A",
            group,
            [award, "B"],
        ), res
        assert abs(res["steps"] * DT - (limit + DT)) < 1e-6


def test_exploit_rocking_across_a_rod_border_does_not_reset_the_clock():
    """A ball kept moving in the band both 5-rods reach stays in both rods' possession: the clocks do not restart."""
    r = Foosball(duel=True)
    open_play(r)
    for k in range(2000):
        place(r, 0.0015 if k % 20 < 10 else -0.0015, 0.1)
        r.step()
        if r.results:
            break
    assert r.results and r.results[-1]["steps"] * DT <= foos.LIMIT_S["midfield"] + 0.02


def test_exploit_parking_the_ball_in_a_corner_is_a_foul():
    r = Foosball(duel=True)
    open_play(r)
    place(r, -0.58, -0.30)
    while not r.results:
        r.step()
    res = r.results[-1]
    assert (res["how"], res["guilty"], res["group"]) == ("possession_time", "A", "goalie_rods")


def test_exploit_ball_cannot_get_stuck():
    """With nobody playing, from every restart, a rally ends within the longest possession limit plus the roll there;
    every playfield point is within reach of some rod (no dead zone)."""
    for first in "AB":
        r = Foosball(duel=True, first=first)
        play_out(r, max_steps=round(80 / DT))
        assert r.results
        assert all(x["steps"] * DT <= max(foos.LIMIT_S.values()) + 5 for x in r.results if x["how"] != "time_cap")
    r = Foosball(duel=True)
    for x in np.linspace(-0.58, 0.58, 59):
        for y in np.linspace(-0.31, 0.31, 32):
            assert r.groups_reaching(np.array([x, y, foos.BALL_R])), (x, y)


def test_exploit_passive_play_loses():
    """Doing nothing loses to the scripted team (format match, through the plugin); a team that only blocks (never
    shoots from the 5-rod or the 3-rod) loses to one that plays."""
    test_reference_policy_is_the_opponents()
    m = SPORT.build(
        MatchConfig(sport="foosball", format="match", seats=("A",), seed=1, first="A", action_level="low", level=1)
    )
    while m.phase().state != "finished":
        if m.phase().state == "unit_over":
            m.resume()
        m.step()
    from core.outcome import decide

    out = decide(m, None, None, ["A"])
    assert out.winner == "opponent", (out, m.scores())
    r = Foosball(duel=True, first="A", seed=11)
    play_out(r, {"A": dict(style="solo", shoot=False), "B": dict(style="team")})
    assert r.done and r.winner() == "B"


def test_exploit_bad_and_frozen_actions_are_refused():
    m, layer = SPORT.build(DUEL), SPORT.action_layer("low")
    d0 = m.state_digest()
    for body in (
        {},
        {"rods": {}},
        {"rods": {"striker": {"slide": 0}}},
        {"rods": {"goalie": {}}},
        {"rods": {"goalie": {"omega": 5}}},
        {"rods": {"goalie": {"slide": 0.12}}},
        {"rods": {"midfield": {"slide": -0.07}}},
        {"rods": {"attack": {"angle": 2.7}}},
        {"rods": {"attack": {"angle": float("nan")}}},
        {"rods": {"attack": {"angle": True}}},
        {"rods": {"attack": {"angle": 1, "omega": 81}}},
        {"rods": {"attack": {"angle": 1, "omega": 0}}},
        {"rods": {"attack": {"slide": 0, "speed": 2.6}}},
        {"rods": {"attack": {"slide": 0, "kick": 1}}},
        {"rods": {"goalie": {"slide": 0}}, "x": 1},
        {"type": "target", "rods": {"goalie": {"slide": 0}}},
        {"rods": {"attack": {"angle": "1"}}},
        {"rods": [1, 2]},
        {"type": "rods"},
    ):
        try:
            layer.parse("A", body, m, {})
            raise AssertionError(f"accepted {body}")
        except Refused as e:
            assert e.code == 400, (body, e.code)
    assert m.state_digest() == d0 and m.k == 0
    cmd = layer.parse("A", {"rods": {"midfield": {"slide": 0.06, "angle": -1}, "goalie": {"angle": 2.6}}}, m, {})
    assert cmd == {"type": "rods", "rods": {"midfield": {"slide": 0.06, "angle": -1.0}, "goalie": {"angle": 2.6}}}
    assert layer.parse("B", {"type": "rods", "rods": {"attack": {"slide": 0.1}}}, m, {})["rods"] == {
        "attack": {"slide": 0.1}
    }
    r = m.r  # the 20 ms latency: the command reaches the motors 2 steps on
    info = m.apply("A", {"type": "rods", "rods": {"defence": {"slide": 0.19}}})
    assert info == {"taken_step": 0, "effect_step": 2}
    i = r.rod_index("A", "defence")
    m.step()
    m.step()
    assert r.cmd_slide[i] == 0.0
    m.step()
    assert r.cmd_slide[i] == 0.19
    while m.phase().state == "playing":  # nobody plays: a possession foul freezes the world
        m.step()
    d1 = m.state_digest()
    assert "rejected" in m.apply("A", {"type": "rods", "rods": {"attack": {"angle": 1.0}}})
    assert m.state_digest() == d1


def test_exploit_world_state_hidden():
    from core import local

    for cfg in (
        MATCH,
        MatchConfig(sport="foosball", format="duel", seats=("A", "B"), seed=123457, first="B", action_level="low"),
    ):
        m = SPORT.build(cfg)
        for s, (a, b, c, e) in zip(cfg.seats, ((0.03, -0.07, 0.237, -0.419), (-0.041, 0.083, 0.311, -0.523))):
            m.apply(
                s, {"type": "rods", "rods": {"midfield": {"slide": a, "angle": c}, "attack": {"slide": b, "angle": e}}}
            )
        for _ in range(60):
            if m.phase().state == "playing":
                m.step()
        r = m.r
        hidden = [float(x) for x in r.ball()[:2]] + [float(x) for x in r.ball_velocity()[:2]]
        for s in cfg.seats:
            other = OTHER[s]
            hidden_s = hidden + [float(r.data.qpos[r.sq[r.rod_index(other, x)]]) for x in RODS]
            hidden_s += [float(r.data.qpos[r.aq[r.rod_index(other, x)]]) for x in RODS]
            spec = local.seat_spec(SPORT, cfg, s, m, plugin=PLUGIN)
            obs = m.observe(s)
            keys = keys_of([spec, obs])
            assert "seed" not in keys and not any("seed" in k.lower() for k in keys), keys
            assert str(cfg.seed) not in json.dumps([spec, obs], default=float)
            shown = numbers_of([spec, obs])
            for v in hidden_s:
                assert abs(v) <= 1e-3 or not any(abs(n - v) < 1e-9 for n in shown), (s, v)


def test_exploit_review_stalling_forfeits():
    """Nobody plays: the first rally ends (a possession foul) and the world freezes; A reviews, B does not: B loses
    by ready_timeout (valid)."""
    from core.realtime import RealtimeService

    env = {
        "READY_TIMEOUT_S": "1.0",
        "START_TIMEOUT_S": "30",
        "BARRIER_LEAD_S": "0.05",
        "RCB_ACT_RULE": "program",
        "PLAYER_MODE_A": "oracle",
        "PLAYER_MODE_B": "oracle",
        "RCB_BROADCAST": "0",
        "RCB_RENDER_VIDEO": "0",
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        cfg = MatchConfig(
            sport="foosball", format="duel", seats=("A", "B"), seed=1, first="A", action_level="low", slowdown=0.05
        )
        with tempfile.TemporaryDirectory() as tmp:
            svc = RealtimeService(SPORT, PLUGIN, cfg, Path(tmp))
            threading.Thread(target=svc.loop, daemon=True).start()
            post = svc.routes(True)
            th = [threading.Thread(target=post["start"], args=(s, {}), daemon=True) for s in "AB"]
            for t in th:
                t.start()
            for t in th:
                t.join(30)
            t0 = time.time()
            while svc.phase != "review" and time.time() - t0 < 120:
                time.sleep(0.05)
            assert svc.phase == "review" and svc.m.r.results[0]["how"] == "possession_time", svc.phase
            assert post["act"]("A", {"rods": {"attack": {"angle": 1.0}}})[0] == 409  # frozen during the review
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


def test_exploit_no_draw_and_no_spinning():
    """A finished game always has a winner: more goals, else the last goal, else less time with the ball in one's own
    half, else the player who did not have the first kick-off (the tiebreak chain equals foos.Foosball.winner()). The
    rods cannot make a full turn, and a transition after a contact spin is a foul whose goal does not count."""
    m = SPORT.build(
        MatchConfig(sport="foosball", format="duel", seats=("A", "B"), seed=1, first="A", action_level="low")
    )
    r = m.r
    assert [n for n, _ in m.tiebreaks()] == ["goals", "last_goal", "territory", "first_kickoff"]

    def chain():
        return next((w, n) for n, rule in m.tiebreaks() for w in [rule(m)] if w is not None)

    open_play(r)
    r.k = round(foos.CAP_S / DT) - 1
    r.step()
    assert r.done and r.results[-1]["how"] == "time_cap" and m.phase().state == "finished"
    for half, results, want in (
        ({"A": 5, "B": 5}, [], ("B", "first_kickoff")),
        ({"A": 7, "B": 5}, [], ("B", "territory")),
        ({"A": 5, "B": 5}, ["A", "B"], ("B", "last_goal")),
        ({"A": 5, "B": 5}, ["A", "B", "A"], ("A", "goals")),
    ):
        r.half_steps = half
        r.results = [
            dict(rally=i + 1, how="goal", winner=w, guilty=None, next=["kickoff", OTHER[w]], k=i + 1, steps=1)
            for i, w in enumerate(results)
        ]
        assert chain() == want and r.winner() == want[0], (half, results, chain())
    mm = SPORT.build(
        MatchConfig(sport="foosball", format="match", seats=("A",), seed=1, first="A", action_level="low", level=1)
    )
    mm.r.done, mm.r.half_steps = True, {"A": 5, "B": 5}
    assert next(w for _, rule in mm.tiebreaks() for w in [rule(mm)] if w is not None) == "opponent"
    for s in "AB":  # no full turn: stops and command range
        for rod in RODS:
            lo, hi = r.model.jnt_range[r.model.joint(f"{s}_{rod}_spin").id]
            assert hi - lo < 2 * math.pi
    try:
        validate_action({"rods": {"attack": {"angle": 6.4}}})
        raise AssertionError("a full-turn target was accepted")
    except ValueError:
        pass
    g = Foosball(duel=True)  # a contact spin (forced): a foul, the goal does not count
    open_play(g)
    i = g.rod_index("A", "attack")
    g.set_command("A", {"attack": dict(angle=-1.6)})
    for _ in range(20):
        g.step()
    place(g, 0.225 - 0.03, 0.0)
    g.set_command("A", {"attack": dict(angle=1.1)})
    g.sweep[i] = 7.0
    g.sweep_dir[i] = 1.0
    orig = g.track_spin
    g.track_spin = lambda: (orig(), g.sweep.__setitem__(i, g.sweep[i] + 7.0) if g.sweep_dir[i] > 0 else None)
    while not g.results:
        g.step()
    res = g.results[-1]
    assert (res["how"], res["guilty"], res["next"]) == ("illegal_spin", "A", ["midfield", "B"]), res
    assert g.scores()["A"]["points"] == 0
