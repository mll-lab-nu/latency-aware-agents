"""curling's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py, after
the generic suite). Run alone: python3 -m core.testkit tests sports/curling/plugin.py (in the sport's image)."""
import os
import tempfile
from pathlib import Path

from core.episode import load_plugin
from core.outcome import decide
from core.sport import MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
DUEL = MatchConfig(sport="curling", format="duel", seats=("A", "B"), seed=1, first="A", action_level="low")
MATCH = MatchConfig(sport="curling", format="match", seats=("A",), seed=1, first="A", action_level="low", level=1)
DRAW = {"type": "throw", "speed": 2.39, "angle": 3.1, "spin": "ccw"}

import curling as C  # noqa: E402  (the plugin's directory is on sys.path now)


def throw(m, seat, body, clocks=None):
    layer = SPORT.action_layer("low")
    cmd = layer.parse(seat, body, m, {"clock_left_s": clocks or {s: 3600.0 for s in m.cfg.seats}})
    return m.apply(seat, layer.lower(seat, cmd, None, m)[0])


def test_exploit_bad_throws_are_refused():
    for cfg in (DUEL, MATCH):
        m, layer = SPORT.build(cfg), SPORT.action_layer("low")
        d0 = m.state_digest()
        for body in ({**DRAW, "speed": 4.5}, {**DRAW, "speed": 0.2}, {**DRAW, "angle": 7.0}, {**DRAW, "spin": "left"},
                     {**DRAW, "spin": "CCW"}, {"type": "throw", "speed": 2.4, "angle": 0.0},
                     {**DRAW, "speed": float("nan")}, {**DRAW, "angle": float("inf")}, {**DRAW, "sweep": True},
                     {**DRAW, "type": "sweep"}, {**DRAW, "speed": "2.4"}):
            try:
                layer.parse("A", body, m, {"clock_left_s": {}})
                raise AssertionError(f"accepted {body}")
            except Refused as e:
                assert e.code == 400
        assert m.state_digest() == d0 and m.k == 0


def test_exploit_out_of_turn():
    from core.turn import TurnService
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    with tempfile.TemporaryDirectory() as tmp:
        svc = TurnService(SPORT, PLUGIN, DUEL, Path(tmp))
        post = svc.routes(True)
        assert post["act"]("A", DRAW)[0] == 409                       # before the start: refused
        post["start"]("A", {})
        post["start"]("B", {})
        assert post["act"]("B", DRAW)[0] == 409                       # A throws first (FIRST_PLAYER = A)
        code, out = post["act"]("A", DRAW)
        assert code == 200 and svc.m.k == 1 and out["your_throw"]["by"] == "you"
        assert post["act"]("A", DRAW)[0] == 409                       # the review comes first
        assert post["ready"]("A", {"why": "w", "plan": "p"})[0] == 200
        assert post["act"]("A", DRAW)[0] == 409                       # now it is B's turn
        assert svc.m.k == 1
        assert post["act"]("B", DRAW)[0] == 200 and svc.m.k == 2
    m = SPORT.build(DUEL)                                           # the world itself refuses too
    try:
        m.apply("B", {"type": "throw", "speed": 2.4, "angle": 3.0, "spin": "ccw"})
        raise AssertionError("B threw out of turn")
    except Refused as e:
        assert e.code == 409


def test_exploit_no_positions_in_observation():
    for cfg in (DUEL, MATCH):
        m = SPORT.build(cfg)
        for i in range(6):
            if m.to_move() is None:
                break
            throw(m, m.to_move(), {**DRAW, "angle": 2.5 + 0.2 * i})
        assert m.g.sheet.stones, "no stone in play"
        text = "".join(repr(m.observe(s)) for s in cfg.seats)
        for s in m.g.sheet.stones:
            for x in (s["x"], s["y"]):
                assert repr(x) not in text and f"{x:.3f}" not in text and f"{x:.2f}" not in text, x


def test_exploit_free_guard_zone():
    m = SPORT.build(DUEL)
    v, a = C.master("ccw").aim(0.0, 35.0, 0.0)                     # A: a centre guard in the free guard zone
    throw(m, "A", {"type": "throw", "speed": round(v, 6), "angle": round(a, 6), "spin": "ccw"})
    guard = dict(next(s for s in m.g.sheet.stones if s["team"] == 0))
    assert m.g.sheet.in_fgz(guard), guard
    v, a = C.master("ccw").aim(guard["x"], guard["y"], 2.5)        # B (stone 2 of the end): a firm takeout of it
    info = throw(m, "B", {"type": "throw", "speed": round(v, 6), "angle": round(a, 6), "spin": "ccw"})
    rec = info["rec"]
    assert rec["fgz_violation"] and rec["removed"][0][0] == rec["sid"], rec
    left = [s for s in m.g.sheet.stones]
    assert len(left) == 1 and (left[0]["x"], left[0]["y"]) == (guard["x"], guard["y"])   # put back exactly
    assert "free_guard_zone_violation" in SPORT.action_layer("low").reply("B", {**DRAW}, info, m)["your_throw"]


def tie_at_the_last_stone(cfg):
    """A level match before the last stone of the last extra end, with an empty sheet (match: the player has it)."""
    m = SPORT.build(cfg)
    g = m.g
    g.score, g.end, g.k = [2, 2], C.N_ENDS + C.MAX_EXTRA_ENDS - 1, 2 * C.N_STONES - 1
    if cfg.format == "match":
        g.hammer = 0
    return m


def test_exploit_tie_rules():
    short = {"type": "throw", "speed": 1.0, "angle": 0.0, "spin": "ccw"}   # stops short of the hog line: removed
    for first, clocks, want, by in (("A", {"A": 100.0, "B": 50.0}, "A", "chess_clock"),
                                    ("A", {"A": 50.0, "B": 100.0}, "B", "chess_clock"),
                                    ("A", {"A": 70.0, "B": 70.0}, "B", "hammer_in_end_1"),
                                    ("B", {"A": 70.0, "B": 70.0}, "A", "hammer_in_end_1")):
        cfg = MatchConfig(sport="curling", format="duel", seats=("A", "B"), seed=1, first=first, action_level="low")
        m = tie_at_the_last_stone(cfg)
        throw(m, m.to_move(), short, clocks)
        assert m.g.done and m.g.score == [2, 2] and m.g.tiebreak["clock_left_s"] == [clocks["A"], clocks["B"]]
        o = decide(m, sides=("A", "B"))
        assert (o.winner, o.decided_by) == (want, by), (first, clocks, o)
    m = tie_at_the_last_stone(MATCH)                                   # the scripted team wins a tie
    throw(m, "A", short, {"A": 3599.0})
    assert m.g.done and m.g.score == [2, 2] and m.g.tiebreak["winner"] == 1
    o = decide(m, sides=("A",))
    assert (o.winner, o.decided_by) == ("opponent", "chess_clock"), o


def test_exploit_independent_noise_streams():
    """Each team draws its own noise stream: within a match the n-th stones of the two teams in an end scatter
    independently; the other first order of a duel on the same seed swaps the streams (the same luck over the pair)."""
    seen = {}
    for first in ("A", "B"):
        cfg = MatchConfig(sport="curling", format="duel", seats=("A", "B"), seed=7, first=first, action_level="low")
        m = SPORT.build(cfg)
        role = {m.g.first: "first", {"A": "B", "B": "A"}[m.g.first]: "second"}
        while not m.g.done and m.g.end < 2:
            seat = m.to_move()
            team = 0 if seat == "A" else 1
            key = (first, m.g.end, C.N_STONES - m.g.stones_left(team))
            seen.setdefault(key, {})[role[seat]] = m.g._noise(team)
            throw(m, seat, {**DRAW, "angle": 2.0 + 0.1 * m.g.k})
    for (first, end, n), v in seen.items():
        if len(v) == 2:
            assert v["first"] != v["second"], (first, end, n)              # independent within the match
        other = seen.get(({"A": "B", "B": "A"}[first], end, n))
        if other:                                                          # the pair: the same draw for the same role
            assert all(other[r] == v[r] for r in v if r in other), (first, end, n)
    m = SPORT.build(MATCH)
    g = m.g
    assert g.noise_stream(0) == 1 and g.noise_stream(1) == 2
    import numpy as np
    za = [np.random.default_rng([s, e, 1, n]).standard_normal(2)[0] for s in range(1, 21) for e in (1, 2) for n in range(5)]
    zb = [np.random.default_rng([s, e, 2, n]).standard_normal(2)[0] for s in range(1, 21) for e in (1, 2) for n in range(5)]
    assert abs(float(np.corrcoef(za, zb)[0, 1])) < 0.2
