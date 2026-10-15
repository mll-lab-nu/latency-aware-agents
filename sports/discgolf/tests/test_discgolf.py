"""Disc golf's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py after
the generic suite), ported from the Season-1 unit tests (scripts/olympiad/discgolf/test_discgolf.py). Run alone:
python3 -m core.testkit tests sports/discgolf/plugin.py (in the sport's image)."""
import json
import math
import os
import tempfile
import time
import tomllib
from pathlib import Path

from core.episode import load_plugin
from core.sport import FINISHED, MatchConfig, Refused

HERE = Path(__file__).resolve().parents[1]
PLUGIN = str(HERE / "plugin.py")
SPORT = load_plugin(PLUGIN)
DECL = tomllib.loads((HERE / "sport.toml").read_text())
ZERO = {k: 0.0 for k in ("aim_deg", "launch_deg", "nose_deg", "bank_deg", "speed_frac", "spin_frac")}


def cfg(seed=1):
    return MatchConfig(sport="discgolf", format="single", seats=("A",), seed=seed, first="A", action_level="low")


def body(**kw):
    b = dict(type="throw", disc="midrange", speed=20.0, launch_deg=10.0, nose_deg=0.0, bank_deg=0.0, spin_rps=12.0,
             aim_deg=0.0)
    b.update(kw)
    return b


def cmd(m, clocks=None, **kw):
    """A low-level command as the action layer makes it (schema, then the legacy validate())."""
    return SPORT.action_layer("low").parse("A", body(**kw), m, {"clock_left_s": clocks or {"A": 3600.0}})


def service(tmp):
    from core.turn import TurnService
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    return TurnService(SPORT, PLUGIN, cfg(), Path(tmp))


def finish(svc):
    for _ in range(600):                                             # the fresh-process replay, before the cleanup
        if svc.finalized:
            return
        time.sleep(0.1)


def test_exploit_bad_actions_are_refused():
    m, layer = SPORT.build(cfg()), SPORT.action_layer("low")
    S, D = m.S, m.S.D
    # the declared limits, scatter and success line are the simulator's
    lim = {k: (f["min"], f["max"]) for k, f in m.schema("A")["throw"].items() if k != "disc"}
    assert lim == D.LIMITS and m.schema("A")["throw"]["disc"]["values"] == list(S.FL.DISC_NAMES)
    assert DECL["spec"]["execution_scatter_1sigma"] == D.SCATTER
    assert DECL["formats"]["single"]["success"]["score_at_most"] == S.SUCCESS_MAX_TOTAL
    d0 = m.state_digest()
    for bad in (dict(speed=31.0), dict(speed=4.9), dict(bank_deg=61.0), dict(launch_deg=36.0), dict(nose_deg=-11.0),
                dict(spin_rps=21.0), dict(spin_rps=2.0), dict(spin_rps=-3.9), dict(aim_deg=181.0),
                dict(disc="frisbee"), dict(speed=float("nan")), dict(type="putt"), dict(speed="20"),
                dict(trees=[]), dict(aim_deg=float("inf"))):
        try:
            layer.parse("A", body(**bad), m, {"clock_left_s": {"A": 1.0}})
            raise AssertionError(f"accepted {bad}")
        except Refused as e:
            assert e.code == 400
    try:
        layer.parse("A", {k: v for k, v in body().items() if k != "nose_deg"}, m, {"clock_left_s": {}})
        raise AssertionError("accepted a throw without nose_deg")
    except Refused:
        pass
    assert m.state_digest() == d0 and m.k == 0
    c = layer.parse("A", body(speed=20.123456, disc="driver"), m, {"clock_left_s": {"A": 12.34567}})
    assert c["speed"] == 20.1235 and c["disc_id"] == 2 and c["clock_left_s"] == {"A": 12.346}
    assert layer.parse("A", body(), m, {"clock_left_s": {"A": -1}})["clock_left_s"] == {"A": 0.0}


def test_exploit_out_of_turn():
    with tempfile.TemporaryDirectory() as tmp:
        svc = service(tmp)
        post = svc.routes(True)
        b = body(disc="putter", speed=14.0, launch_deg=8.0)
        assert post["act"]("A", b)[0] == 409                         # before the start: refused
        post["start"]("A", {})
        code, out = post["act"]("A", b)
        assert code == 200 and svc.m.k == 1 and "status" in out["result"]
        assert post["act"]("A", b)[0] == 409                         # the review comes first
        assert svc.m.k == 1
        assert post["ready"]("A", {"why": "w", "plan": "p"})[0] == 200
        assert post["act"]("A", b)[0] == 200 and svc.m.k == 2


def test_exploit_stalling_loses():
    with tempfile.TemporaryDirectory() as tmp:                       # abandoned: a valid failure
        svc = service(tmp)
        post = svc.routes(True)
        post["start"]("A", {})
        post["act"]("A", body(disc="putter", speed=14.0, launch_deg=8.0))
        with svc.cond:
            svc.lose("A", "abandoned")
        o = svc.outcome()
        assert svc.reason == "abandoned" and o.winner is None and svc.m.phase().state != FINISHED
        finish(svc)
    with tempfile.TemporaryDirectory() as tmp:                       # the clock ends the round: par + 4 each
        svc = service(tmp)
        post = svc.routes(True)
        post["start"]("A", {})
        post["act"]("A", body(disc="putter", speed=14.0, launch_deg=8.0))
        with svc.cond:
            svc.lose("A", "clock")
        o = svc.outcome()
        D = svc.m.S.D
        assert svc.reason == "clock" and o.winner is None and svc.m.phase().state != FINISHED
        ff = svc.m.final_fields()
        assert ff["clock_out"]["A"] is True and not ff["complete"]["A"]
        assert ff["scores"]["A"]["holes"][1:] == [H["par"] + D.MAX_OVER_PAR + 1 for H in D.HOLES[1:]]
        assert svc.m.scores()["A"] >= D.PAR_TOTAL + 4 * (D.N_HOLES - 1)
        finish(svc)


def test_exploit_no_world_state():
    c = cfg(seed=987654)
    m = SPORT.build(c)
    m.apply("A", cmd(m, disc="driver", speed=24.0, aim_deg=3.0))
    m.apply(m.to_move(), cmd(m, disc="driver", speed=22.0, aim_deg=-2.0))
    obs = json.dumps(m.observe("A"), default=float)
    text = obs + json.dumps(SPORT.spec(c, "A", m), default=float)
    assert "987654" not in text and '"seed"' not in text
    for key in ("tree_list", "trees", "trail", "track", "rest_exact", "release", "wind", "executed", "scatter_draw"):
        assert f'"{key}"' not in obs, key
    D = m.S.D
    cur = m.M._cur("A")
    w = D.wind_of(cur.key(), cur.throws[cur.i])
    for v in w:                                                      # the wind of the next throw
        assert f"{v:.3f}" not in obs and f"{v:.4f}" not in obs, v
    trees = [t for H in D.HOLES for t in H["tree_list"]]              # no tree coordinates (forest trees are jittered)
    assert len(trees) > 100
    leaked = [t for t in trees if t["kind"] == "forest" and f"{t['x']!r}, {t['y']!r}" in text]
    assert not leaked, leaked[:3]
    assert '"trees"' not in json.dumps(SPORT.spec(c, "A", m)["match"]["holes"])


def test_exploit_ob_and_mandatory_penalised():
    m = SPORT.build(cfg(seed=m_seed()))
    S, D = m.S, m.S.D
    c = D.Course()                                                   # hole 4: the pond in front of the basket
    c.i, c.hole, c.lie = 3, 3, (0.0, 0.0)
    rec, _ = c.play(S.validate(body(disc="putter", speed=16.0, launch_deg=8.0, aim_deg=-3.5)))
    assert rec["status"] == "out of bounds" and rec["penalty"] == 1 and c.count(3) == 2, rec
    assert D.in_bounds(D.HOLES[3], *c.lie) and c.lie[0] < 32.5, c.lie
    H = D.HOLES[4]                                                   # hole 5: the mandatory, passed on its right
    hits = 0
    for aim in (6.0, 8.0, 10.0):
        c2 = D.Course()
        c2.i, c2.hole, c2.lie = 4, 4, (0.0, 0.0)
        rec, _ = c2.play(S.validate(body(disc="midrange", speed=22.0, aim_deg=aim)))
        if rec.get("mandatory") == "missed":
            hits += 1
            assert rec["status"] == "missed the mandatory" and rec["penalty"] == 1
            assert tuple(rec["next_lie"]) == tuple(H["mando"]["dz"])
    assert hits >= 2


def m_seed():
    return 20261005                                                  # the standard course (the legacy tests' course)


def test_exploit_cap_fast_basket_and_trees():
    m = SPORT.build(cfg(seed=m_seed()))
    S, D = m.S, m.S.D
    back = cmd(m, disc="putter", speed=5.0, launch_deg=-5.0, aim_deg=180.0)          # backwards, 5 m/s
    par = D.HOLES[0]["par"]
    for _ in range(par + D.MAX_OVER_PAR):
        res = m.apply("A", back)["result"]
    assert res["hole_finished"] and res["hole_score"] == par + D.MAX_OVER_PAR + 1
    assert m.observe("A")["decision"]["hole"] == 2
    H = D.HOLES[0]
    bx, by = H["basket"]
    lie = (bx - 6.0, by)
    best = None
    for spd in (8.0, 9.0, 10.0, 11.0, 12.0):                         # a putt that is caught
        for la in range(-4, 12):
            r = D.throw(H, lie, dict(disc="putter", speed=spd, launch_deg=float(la), nose_deg=0.0, bank_deg=0.0,
                                     spin_rps=10.0, aim_deg=0.0), (0.0, 0.0), ZERO, trail=False)
            if r["holed"]:
                best = (spd, la)
                break
        if best:
            break
    assert best, "no putt from 6 m is caught"
    fast = D.throw(H, lie, dict(disc="putter", speed=29.0, launch_deg=best[1] * 0.3, nose_deg=-2.0, bank_deg=0.0,
                                spin_rps=10.0, aim_deg=0.0), (0.0, 0.0), ZERO, trail=False)
    assert not fast["holed"], fast["result"]
    H = D.HOLES[1]                                                   # the left dogleg: the woods on the direct line
    bx, by = H["basket"]
    r = D.throw(H, (0.0, 0.0), dict(disc="midrange", speed=24.0, launch_deg=10.0, nose_deg=0.0, bank_deg=0.0,
                                    spin_rps=12.0, aim_deg=math.degrees(math.atan2(by, bx))), (0.0, 0.0), ZERO,
                trail=False)
    assert r["hits"]["canopy"] + r["hits"]["trunk"] >= 1 and math.hypot(r["rest"][0] - bx, r["rest"][1] - by) > 20
