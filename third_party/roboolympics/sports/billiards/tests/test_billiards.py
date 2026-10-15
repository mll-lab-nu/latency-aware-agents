"""billiards' own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py, after
the generic suite). Run alone: python3 -m core.testkit tests sports/billiards/plugin.py (in the sport's image)."""
import os
import tempfile
from pathlib import Path

from core.episode import load_plugin
from core.outcome import decide
from core.sport import MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
DUEL = MatchConfig(sport="billiards", format="duel", seats=("A", "B"), seed=1, first="B", action_level="low")
# (the coin toss of seed 1 falls 1: the other player, A, breaks)
SINGLE = MatchConfig(sport="billiards", format="eightball", seats=("A",), seed=1, first="A", action_level="low",
                     level=1)
BREAK = {"type": "shoot", "aim_deg": 90.0, "speed": 6.0, "cue_x": 0.4953, "cue_y": 0.4453}
REVIEW = {"why": "w", "plan": "p"}

import eightball as E8  # noqa: E402  (the plugin's directory is on sys.path now)
import render  # noqa: E402
PLUGIN_GLOBALS = type(SPORT).build.__globals__                    # the loaded plugin module's namespace


def shoot(m, seat, body, clocks=None):
    layer = SPORT.action_layer("low")
    cmd = layer.parse(seat, body, m, {"clock_left_s": clocks or {s: 3600.0 for s in m.cfg.seats}})
    return m.apply(seat, layer.lower(seat, cmd, None, m)[0])


def refused(m, seat, body, code=400):
    layer = SPORT.action_layer("low")
    try:
        layer.parse(seat, body, m, {"clock_left_s": {}})
    except Refused as e:
        assert e.code == code, (body, e.code, str(e))
        return str(e)
    raise AssertionError(f"accepted {body}")


def after_break(cfg=DUEL):
    m = SPORT.build(cfg)
    shoot(m, m.to_move(), BREAK)
    return m


def test_exploit_bad_shots_are_refused():
    for cfg in (DUEL, SINGLE):
        m = SPORT.build(cfg)
        d0 = m.state_digest()
        for body in ({**BREAK, "speed": 7.5}, {**BREAK, "speed": 0.1}, {**BREAK, "spin_x": 0.4, "spin_y": 0.4},
                     {**BREAK, "spin_x": 0.6}, {**BREAK, "speed": float("nan")}, {**BREAK, "aim_deg": float("inf")},
                     {**BREAK, "aim_deg": "90"}, {**BREAK, "aim_deg": 400.0}, {**BREAK, "jump": True},
                     {**BREAK, "type": "massé"}, {"type": "shoot", "speed": 2.0},
                     {k: v for k, v in BREAK.items() if k != "cue_y"}, {**BREAK, "cue_x": 2.0},
                     {**BREAK, "call": "3:xx"}, {**BREAK, "call": 3}):
            refused(m, "A", body)
        assert m.state_digest() == d0 and m.k == 0
    m = after_break()                                              # a call is normalized, a bad one refused
    seat = m.to_move()
    refused(m, seat, {"type": "shoot", "aim_deg": 10.0, "speed": 2.0, "call": "16:lb"})
    cmd = SPORT.action_layer("low").parse(seat, {"aim_deg": -90.0, "speed": 2.0, "call": " Safety "}, m, {})
    assert cmd["call"] == "safety" and cmd["aim_deg"] == 270.0 and "cue_x" not in cmd


def test_exploit_out_of_turn():
    from core.turn import TurnService
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    with tempfile.TemporaryDirectory() as tmp:
        svc = TurnService(SPORT, PLUGIN, DUEL, Path(tmp))
        post = svc.routes(True)
        assert post["act"]("A", BREAK)[0] == 409                      # before the start: refused
        post["start"]("A", {})
        post["start"]("B", {})
        assert post["act"]("B", BREAK)[0] == 409                      # A breaks (seed 1's toss: not FIRST_PLAYER B)
        code, out = post["act"]("A", BREAK)
        assert code == 200 and svc.m.k == 1 and out["your_shot"]["player"] == "you", out
        keeps = out["your_shot"]["keeps_turn"]
        nxt = "A" if keeps else "B"
        other = "B" if keeps else "A"
        assert post["act"]("A", BREAK)[0] == 409                      # the review comes first
        assert post["ready"]("A", REVIEW)[0] == 200
        safety = {"type": "shoot", "aim_deg": 45.0, "speed": 2.0, "call": "safety"}
        assert post["act"](other, safety)[0] == 409                   # not this seat's turn
        assert svc.m.k == 1
        assert post["act"](nxt, safety if svc.m.g.break_done else BREAK)[0] == 200 and svc.m.k == 2
    m = SPORT.build(DUEL)                                           # the world itself refuses too
    try:
        m.apply("B", {**BREAK})
        raise AssertionError("B shot out of turn")
    except Refused as e:
        assert e.code == 409
    # who breaks is the seed's coin toss, not the configuration: the first order only names who gets the toss's 0,
    # the two orders of a seed mirror each other, and over seeds the side named first breaks about half the time
    toss = {}
    for seed in range(1, 41):
        b = {f: SPORT.build(MatchConfig(sport="billiards", format="duel", seats=("A", "B"), seed=seed, first=f,
                                        action_level="low")).to_move() for f in ("A", "B")}
        assert b["A"] != b["B"], (seed, b)
        toss[seed] = 0 if b["A"] == "A" else 1
    assert toss[1] == 1 and toss[2] == 0 and toss[3] == 0 and 10 <= sum(toss.values()) <= 30, toss


def test_exploit_no_positions_in_observation():
    for cfg in (DUEL, SINGLE):
        m = SPORT.build(cfg)
        for i in range(4):
            seat = m.to_move()
            if seat is None:
                break
            body = BREAK if not m.g.break_done else {"type": "shoot", "aim_deg": 30.0 + 40 * i, "speed": 2.5,
                                                     "call": "safety"}
            shoot(m, seat, body)
        text = "".join(repr(m.observe(s)) for s in cfg.seats)
        for k, (x, y) in m.g.pos.items():
            for v in (x, y):
                assert repr(v) not in text and f"{v:.3f}" not in text and f"{v:.2f}" not in text, (k, v)
        d0 = m.state_digest()
        seat = cfg.seats[0]
        for q in ({"aim": 12.5}, {"aim": -400.0}, None, {}):
            job, meta = m.camera_job(seat, q)
            assert meta["tiles"] == PLUGIN_GLOBALS["TILES"] and len(meta["cameras"]) == 2
            img = SPORT.worker_call(None, "render", job)
            assert img["png"][:8] == b"\x89PNG\r\n\x1a\n"
        assert meta["sight_aim_deg"] == round(render.default_sight_phi(m.g.pos), 3)
        assert m.camera_job(seat, {"aim": 12.5})[1]["sight_aim_deg"] == 12.5
        try:
            m.camera_job(seat, {"aim": "north"})
            raise AssertionError("a non-numeric aim was accepted")
        except Refused:
            pass
        assert m.state_digest() == d0


def test_exploit_rules_refused_before_the_shot():
    m = SPORT.build(DUEL)
    assert "not called" in refused(m, "A", {**BREAK, "call": "1:lb"})                  # the break is not called
    assert "head string" in refused(m, "A", {**BREAK, "cue_y": 0.9})                   # in hand above the head string
    m = after_break()
    seat = m.to_move()
    assert "called" in refused(m, seat, {"type": "shoot", "aim_deg": 10.0, "speed": 2.0})   # every later shot
    assert "8" in refused(m, seat, {"type": "shoot", "aim_deg": 10.0, "speed": 2.0, "call": "8:lb"})  # open table
    if m.g.ball_in_hand is None:
        assert "ball in hand" in refused(m, seat, {"type": "shoot", "aim_deg": 10.0, "speed": 2.0, "call": "safety",
                                                   "cue_x": 0.3, "cue_y": 0.3})
    assert m.k == 1


def tie_at_the_cap(cfg):
    """Both players one shot short of the limit, the table open (7 - 7), the shooter about to play a safety."""
    m = SPORT.build(cfg)
    g = m.g
    g.break_done, g.ball_in_hand = True, None
    g.n_shots = {0: E8.MAX_SHOTS, 1: E8.MAX_SHOTS}
    g.n_shots[g.turn] = E8.MAX_SHOTS - 1
    return m


def safe_miss(m):
    """A soft safety that cannot pot anything: towards the head rail from the cue ball."""
    return {"type": "shoot", "aim_deg": 270.0, "speed": 0.4, "call": "safety"}


def test_exploit_tie_rules():
    for first, clocks, want, by in (("A", {"A": 100.0, "B": 50.0}, "A", "chess_clock"),
                                    ("A", {"A": 50.0, "B": 100.0}, "B", "chess_clock"),
                                    ("A", {"A": 70.0, "B": 70.0}, "A", "did_not_break"),    # seed 1: B breaks
                                    ("B", {"A": 70.0, "B": 70.0}, "B", "did_not_break")):   # seed 1: A breaks
        cfg = MatchConfig(sport="billiards", format="duel", seats=("A", "B"), seed=1, first=first,
                          action_level="low")
        m = tie_at_the_cap(cfg)
        shoot(m, m.to_move(), safe_miss(m), clocks)
        assert m.g.done and m.g.result.startswith("shot limit: balls left in own group 7 - 7"), m.g.result
        assert m.g.tiebreak["clock_left_s"] == [clocks["A"], clocks["B"]]
        o = decide(m, sides=("A", "B"))
        assert (o.winner, o.decided_by) == (want, by), (first, clocks, o)
    m = tie_at_the_cap(SINGLE)                                     # the scripted opponent wins a tie
    shoot(m, "A", safe_miss(m), {"A": 3599.0})
    assert m.g.done and m.g.tiebreak["winner"] == 1
    o = decide(m, sides=("A",))
    assert (o.winner, o.decided_by) == ("opponent", "chess_clock"), o
    m = tie_at_the_cap(DUEL)                                       # not a tie: fewer balls left wins
    m.g.group = {0: "solids", 1: "stripes"}
    for b in E8.SOLIDS[:3]:
        m.g.pos.pop(b)
    shoot(m, m.to_move(), safe_miss(m))
    o = decide(m, sides=("A", "B"))
    assert (o.winner, o.decided_by) == ("A", "balls_left") and getattr(m.g, "tiebreak", None) is None, o


def test_exploit_same_noise_both_seats():
    """The n-th shot of a game gets the same execution draw whoever shoots it (default_rng([seed, 7919, k])), and the
    two seats see the same thing: swapping the seats and the first order gives the mirrored game."""
    res = []
    for first in ("A", "B"):
        cfg = MatchConfig(sport="billiards", format="duel", seats=("A", "B"), seed=5, first=first, action_level="low")
        m = SPORT.build(cfg)
        seen = []
        for i in range(8):
            seat = m.to_move()
            if seat is None:
                break
            body = BREAK if not m.g.break_done else {"type": "shoot", "aim_deg": 15.0 + 37.0 * i, "speed": 2.2,
                                                     "call": "safety"}
            role = "breaker" if seat == m.d.first else "other"
            seen.append((role, m.observe(seat)))
            shoot(m, seat, body)
        res.append((seen, m.scores(), [dict(r, player=None) for r in m.g.shots]))
    (s1, sc1, r1), (s2, sc2, r2) = res
    assert s1 == s2 and r1 == r2 and sc1 == {"A": sc2["B"], "B": sc2["A"]}, (sc1, sc2)


def test_exploit_engine_failure_is_deterministic():
    """A shot pooltool cannot resolve: here the engine is made to fail on the first simulation of the game's second
    shot. The shot is simulated again with the aim nudged (+0.001 deg), the record says so, nothing else changes, and
    a second match with the same commands gets the same state."""
    real = E8.simulate
    out = []
    for _ in range(2):
        calls = {"n": 0}

        def flaky(pos, phi, V, a, b, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise AssertionError("v_n_0 < 0")
            return real(pos, phi, V, a, b, **kw)

        E8.simulate = flaky
        try:
            m = SPORT.build(DUEL)
            shoot(m, "A", BREAK)
            seat = m.to_move()
            info = shoot(m, seat, {"type": "shoot", "aim_deg": 60.0, "speed": 2.0, "call": "safety"})
        finally:
            E8.simulate = real
        rec = info["rec"]
        assert rec["physics_retry"]["aim_nudge_deg"] == 0.001 and "v_n_0" in rec["physics_retry"]["engine_error"]
        g = m.g
        assert g.k == 2 and sum(g.n_shots.values()) == 2 and len(g.shots) == 2 and len(g.decisions) == 2
        assert abs(g.decisions[1]["aim_deg"] - 60.001) < 1e-9
        out.append(m.state_digest())
    assert out[0] == out[1]
    m = SPORT.build(DUEL)                                          # an engine that never resolves: a clean error
    E8.simulate = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("always"))
    try:
        d0 = m.state_digest()
        try:
            shoot(m, "A", BREAK)
            raise AssertionError("an unresolvable shot was played")
        except RuntimeError as e:
            assert "could not resolve" in str(e)
        assert m.state_digest() == d0 and m.g.k == 0 and not m.g.decisions
    finally:
        E8.simulate = real
