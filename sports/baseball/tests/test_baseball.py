"""baseball's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py, after
the generic suite). Run alone: python3 -m core.testkit tests sports/baseball/plugin.py (in the sport's image)."""
import json
import tomllib
from pathlib import Path

from core.episode import load_plugin
from core.sport import FINISHED, UNIT_OVER, MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
DECL = tomllib.loads((Path(PLUGIN).parent / "sport.toml").read_text())
BAT = MatchConfig(sport="baseball", format="batting", seats=("A",), seed=301, first="A", action_level="low", level=1)
DUEL = MatchConfig(sport="baseball", format="duel", seats=("A", "B"), seed=1, first="A", action_level="low")
SWING = {"type": "swing", "t": 0.3, "plane": 4.0, "height": 0.7, "speed": 34.0, "reach": 0.0}
PITCH = {"type": "pitch", "pitch": "fastball", "speed": 38.0, "x": 0.0, "z": 0.8}

import bb  # noqa: E402  (the plugin's directory is on sys.path now)
import bbduel  # noqa: E402
import bbrt  # noqa: E402


def parse(m, seat, body):
    return SPORT.action_layer("low").parse(seat, body, m, {})


def refused(m, seat, body):
    try:
        parse(m, seat, body)
    except Refused as e:
        return e.code == 400
    return False


def run_to(m, cond, limit=2000):
    for _ in range(limit):
        if cond(m):
            return
        if m.phase().state == UNIT_OVER:
            m.resume()
            continue
        if m.phase().state == FINISHED:
            raise AssertionError("the match ended first")
        m.step()
    raise AssertionError("condition never reached")


def test_exploit_bad_commands_are_refused():
    m = SPORT.build(BAT)
    d0 = m.state_digest()
    for body in ({**SWING, "t": 0.0}, {**SWING, "t": 2.6}, {**SWING, "plane": 36.0}, {**SWING, "height": 0.2},
                 {**SWING, "speed": 35.0}, {**SWING, "reach": 0.3}, {**SWING, "t": float("nan")},
                 {**SWING, "speed": "30"}, {**SWING, "height": True}, {"type": "swing", "t": 0.3}, {**SWING, "bat": 1},
                 {"type": "bunt"}, PITCH, {"t": 0.3}, [1, 2]):
        assert refused(m, "A", body), body
    assert m.state_digest() == d0 and m.k == 0
    d = SPORT.build(DUEL)                                  # A bats first, B pitches
    d0 = d.state_digest()
    for seat, body in (("A", PITCH), ("B", SWING), ("B", {**PITCH, "speed": 30.0}), ("B", {**PITCH, "pitch": "knuckle"}),
                       ("B", {**PITCH, "x": 0.7}), ("B", {**PITCH, "z": 0.1}), ("B", {**PITCH, "speed": float("inf")}),
                       ("A", {**SWING, "t": -1.0})):
        assert refused(d, seat, body), (seat, body)
    assert parse(d, "B", PITCH) and parse(d, "A", SWING) and parse(d, "A", {"type": "none"})
    assert d.state_digest() == d0 and d.k == 0


def test_exploit_commands_lock():
    m = SPORT.build(BAT)
    info = m.apply("A", parse(m, "A", SWING))              # the windup: refused by the rig, nothing changes
    assert not info["accepted"] and "rejected" in info and m.g.plan is None
    run_to(m, lambda x: x.g.released())
    info = m.apply("A", parse(m, "A", {**SWING, "t": 0.05}))   # square sooner than the swing takes: it starts now
    assert info["accepted"] and info.get("swing_started")
    rec = dict(m.g.rec)
    for body in (SWING, {"type": "none"}):
        info = m.apply("A", parse(m, "A", body))
        assert not info["accepted"] and m.g.rec["decision"] == rec["decision"]
    d = SPORT.build(DUEL)
    assert d.apply("B", parse(d, "B", PITCH))["accepted"]
    run_to(d, lambda x: x.g.released())
    info = d.apply("B", parse(d, "B", {**PITCH, "x": 0.3}))
    assert not info["accepted"] and d.g.pitch.target[0] != 0.3 and d.g.pitch_cmd[0]["x"] == 0.0


def test_exploit_no_ball_state_in_observation():
    allowed = set(DECL["observation"]["fields"])
    for cfg, seats in ((BAT, ("A",)), (DUEL, ("A", "B"))):
        m = SPORT.build(cfg)
        if not m.single:
            m.apply("B", parse(m, "B", PITCH))
        run_to(m, lambda x: x.g.released() and x.g.t > 0.2)
        for s in seats:
            o = m.observe(s)
            assert set(o) <= allowed, set(o) - allowed
            assert o["last_pitch"] is None                 # the first pitch is not over yet
            own = ("last_pitch", "command") if o.get("role") == "pitching" else ("last_pitch",)   # its own pitch
            o = {k: v for k, v in o.items() if k not in own}
            keys = set(walk(o))
            bad = keys & {"ball", "pos", "position", "vel", "velocity", "spin", "radar", "radar_mps", "pitch_type",
                          "type", "crossing", "plate_crossing", "zone", "in_zone", "target"}
            assert not bad, (bad, o)
            assert "fastball" not in json.dumps(o)


def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)


def test_exploit_pitch_limit():
    assert bb.PA_MAX_PITCHES == bbduel.PA_MAX_PITCHES == 15
    d = SPORT.build(DUEL)                                  # B never pitches: an automatic ball each time
    d.g.history = [{"pa": 0} for _ in range(14)]           # 14 earlier pitches of this plate appearance (fouls)
    run_to(d, lambda x: x.g.over)
    assert d.g.pa_results[-1]["result"] == "out (pitch limit)" and d.g.score == {"A": 0, "B": 0} and d.g.pa == 1
    g = bb.Game(1, {"seed": 301})
    g.history = [{"pa": 0} for _ in range(14)]
    g.decide({"type": "take"}, 0.0)
    assert g.pa_results[-1]["result"] == "out (pitch limit)" and g.bases == 0 and g.pa == 1


def test_exploit_no_pitch_is_a_ball():
    d = SPORT.build(DUEL)
    run_to(d, lambda x: len(x.g.pa_results) == 1)
    assert [h["outcome"] for h in d.g.history] == ["ball (no pitch)"] * 4
    assert d.g.pa_results[0]["result"] == "walk" and d.g.score["A"] == 1


def test_exploit_taking_every_pitch():
    best = {}
    for level, seeds in ((1, [301, 311, 321]), (2, [302, 312, 322]), (3, [303, 313, 323]), (4, [304, 314, 324])):
        for seed in seeds:
            g = bb.Game(level, {"seed": seed})
            while not g.done:
                g.decide({"type": "take"}, 0.0)
            best[(level, seed)] = g.bases
            assert g.bases < g.lv["threshold"], (level, seed, g.bases, [r["result"] for r in g.pa_results])


def test_exploit_independent_scatter_streams():
    """Within a duel the two batters' plate appearances draw independent streams (windup, pitch and swing scatter:
    slot 1 = the first batter, 2 = the other); the other first order on the same seed swaps them."""
    a, b = SPORT.build(DUEL), SPORT.build(MatchConfig(**{**DUEL.to_json(), "seats": ("A", "B"), "first": "B"}))
    assert a.g.slot("A") == 1 and a.g.slot("B") == 2 and b.g.slot("A") == 2 and b.g.slot("B") == 1
    assert a.g.stream == b.g.stream == 1 and a.g.windup == b.g.windup  # the first batter's first pitch: slot 1 in both
    for m in (a, b):                                       # nobody pitches: walks, the same stream for the first batter
        run_to(m, lambda x: len(x.g.pa_results) == 2 and x.g.over)
    ha, hb = a.g.history, b.g.history
    assert [(h["outcome"], h["j"], h["batter"]) for h in ha] == [
        (h["outcome"], h["j"], {"A": "B", "B": "A"}[h["batter"]]) for h in hb]          # the mirror of each other
    assert a.k == b.k
    w = [(bbduel.windup_s(1, 1, j), bbduel.windup_s(1, 2, j)) for j in range(40)]
    assert sum(x == y for x, y in w) <= 3                  # the two slots' windups are independent draws
    sw = [(bbduel.swing_scatter(1, 1, j), bbduel.swing_scatter(1, 2, j)) for j in range(5)]
    assert all(x["t"] != y["t"] and x["plane"] != y["plane"] for x, y in sw)
    for j in range(5):
        p1, p2 = bbduel.make_pitch(1, 1, j, PITCH), bbduel.make_pitch(1, 2, j, PITCH)
        assert p1.cross != p2.cross                        # the n-th pitch to each batter scatters independently
        assert bbduel.make_pitch(1, 1, j, PITCH).cross == p1.cross   # pure (replays)
    import numpy as np
    z1 = np.array([bbduel.make_pitch(s, 1, j, PITCH).cross[1] for s in range(1, 41) for j in range(5)])
    z2 = np.array([bbduel.make_pitch(s, 2, j, PITCH).cross[1] for s in range(1, 41) for j in range(5)])
    assert abs(float(np.corrcoef(z1, z2)[0, 1])) < 0.2


def test_exploit_spec_is_the_simulator():
    sp = DECL["spec"]
    assert "0.60-1.20 s" in sp["pitch"] and tuple(bbrt.WINDUP) == (0.60, 1.20)
    assert tuple(float(x) for x in sp["pitch"].split("(")[1].split(")")[0].split(",")) == tuple(bb.REL)
    assert "0.05 s after the contact" in sp["pitch"] and bbrt.END_AFTER == 0.05
    assert "15 pitches" in sp["plate_appearance"] and bb.PA_MAX_PITCHES == 15
    assert "3.9741 / SPEED" in sp["swing"] and abs(bbrt.T_SQUARE - 3.9741) < 1e-4
    sw = SPORT.build(BAT).schema("A")["swing"]
    assert (sw["plane"]["min"], sw["plane"]["max"], sw["height"]["min"], sw["height"]["max"], sw["speed"]["min"],
            sw["speed"]["max"], sw["reach"]["min"], sw["reach"]["max"]) == (-15.0, 35.0, 0.30, 1.40, 15.0, 34.0, -0.2, 0.2)
    assert "-15 <= PLANE_DEG <= 35, 0.30 <= HEIGHT <= 1.40 m, 15 <= SPEED <= 34 m/s, -0.20 <= REACH <= 0.20 m" in sp["swing"]
    for name, (lo, hi) in bbduel.ARSENAL.items():
        assert f"{name} {lo:g}-{hi:g}" in sp["duel_pitch"], name
    assert "3 cm on the target and 0.3 m/s" in sp["duel_pitch"] and bbduel.PITCH_SCATTER == dict(target_m=0.03,
                                                                                                speed_mps=0.3)
    assert "bats 5 times" in sp["duel"] and bbduel.PA_PER_SEAT == 5 and bbduel.EXTRA_PAIRS == 5
    assert "mass 0.145 kg, radius 0.0365 m, air density 1.19" in sp["ball"] and "C_D 0.35" in sp["ball"]
    assert (bb.M_BALL, bb.R_BALL, bb.RHO, bb.CD, bb.SPIN_TAU) == (0.145, 0.0365, 1.19, 0.35, 30.0)
    m = SPORT.build(BAT)
    s = SPORT.spec(BAT, "A", m)
    assert s["cameras"] == bb.cam_info() and s["match"]["pa_max_pitches"] == 15
    assert DECL["formats"]["batting"]["success"]["score_at_least"] == bb.LEVELS[1]["threshold"] == 4
    assert all(lv["threshold"] == 4 and lv["n_pa"] == 5 for lv in bb.LEVELS.values())
