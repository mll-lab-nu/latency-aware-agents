"""Mini-golf putting's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py
after the generic suite). Run alone: python3 -m core.testkit tests sports/golf/plugin.py (in the sport's image)."""

import json
import math
import os
import tempfile
import tomllib
from pathlib import Path

from core.episode import load_plugin
from core.sport import FINISHED, MatchConfig, Refused

HERE = Path(__file__).resolve().parents[1]
PLUGIN = str(HERE / "plugin.py")
SPORT = load_plugin(PLUGIN)
DECL = tomllib.loads((HERE / "sport.toml").read_text())


def cfg(seed=1):
    return MatchConfig(sport="golf", format="single", seats=("A",), seed=seed, first="A", action_level="low")


def putt(aim=0.0, speed=2.0, clocks=None):
    return {"type": "putt", "aim_deg": aim, "speed": speed, "clock_left_s": clocks or {"A": 3600.0}}


DRIBBLE = putt(aim=180.0, speed=0.05)  # a stroke that cannot hole: the ball barely moves


def service(tmp):
    from core.turn import TurnService

    os.environ["RCB_PLAYER_MODE"] = "oracle"
    return TurnService(SPORT, PLUGIN, cfg(), Path(tmp))


def finish(svc):
    import time

    for _ in range(600):  # the fresh-process replay, before the cleanup
        if svc.finalized:
            return
        time.sleep(0.1)


def test_exploit_bad_actions_are_refused():
    m, layer = SPORT.build(cfg()), SPORT.action_layer("low")
    S, G = m.S, m.S.G
    # the declared limits are the simulator's validate(), and so are the published scatter, pick-up and success line
    lim = {k: (f["min"], f["max"]) for k, f in m.schema("A")["putt"].items()}
    assert lim == {"aim_deg": (-180.0, 180.0), "speed": (0.05, 6.0)}
    for k, (lo, hi) in lim.items():
        assert isinstance(S.validate({"type": "putt", "aim_deg": 0.0, "speed": 1.0, k: lo}), dict)
        assert isinstance(S.validate({"type": "putt", "aim_deg": 0.0, "speed": 1.0, k: hi}), dict)
    sc = DECL["spec"]["execution_scatter_1sigma"]
    assert (sc["aim_deg"], sc["speed_rel"]) == (G.AIM_SIGMA_DEG, G.SPEED_SIGMA)
    assert (
        DECL["formats"]["single"]["success"]["score_at_most"] == S.SUCCESS_MAX_TOTAL == sum(H["par"] for H in G.HOLES)
    )
    assert G.MAX_STROKES == 6 and S.FORFEIT_SCORE == 7
    good = {"type": "putt", "aim_deg": 10.0, "speed": 1.5}
    d0 = m.state_digest()
    for body in (
        {**good, "speed": 6.5},
        {**good, "speed": 0.01},
        {**good, "aim_deg": 181.0},
        {**good, "speed": float("nan")},
        {**good, "aim_deg": "45"},
        {"type": "putt", "speed": 1.0},
        {**good, "type": "roll"},
        {**good, "ball_x": 1.0},
        {**good, "aim_deg": float("inf")},
        [1, 2],
    ):
        try:
            layer.parse("A", body, m, {"clock_left_s": {"A": 1.0}})
            raise AssertionError(f"accepted {body}")
        except Refused as e:
            assert e.code == 400
    assert m.state_digest() == d0 and m.k == 0
    cmd = layer.parse("A", {**good, "aim_deg": 12.34567, "speed": 1.234567}, m, {"clock_left_s": {"A": 12.34567}})
    assert cmd["aim_deg"] == 12.346 and cmd["speed"] == 1.2346  # the legacy rounding
    assert cmd["clock_left_s"] == {"A": 12.346}
    assert layer.parse("A", good, m, {"clock_left_s": {"A": -1}})["clock_left_s"] == {"A": 0.0}


def test_exploit_out_of_turn():
    with tempfile.TemporaryDirectory() as tmp:
        svc = service(tmp)
        post = svc.routes(True)
        body = {"type": "putt", "aim_deg": 180.0, "speed": 0.3}
        assert post["act"]("A", body)[0] == 409  # before the start: refused
        post["start"]("A", {})
        code, out = post["act"]("A", body)
        assert code == 200 and svc.m.k == 1 and out["result"]["result"] == "stopped"
        assert post["act"]("A", body)[0] == 409  # the review comes first
        assert svc.m.k == 1
        assert post["ready"]("A", {"why": "w", "plan": "p"})[0] == 200
        assert post["act"]("A", body)[0] == 200 and svc.m.k == 2


def test_exploit_stalling_loses():
    for why in ("clock", "abandoned"):  # a round ended early is a valid failure
        with tempfile.TemporaryDirectory() as tmp:
            svc = service(tmp)
            post = svc.routes(True)
            post["start"]("A", {})
            post["act"]("A", {"type": "putt", "aim_deg": 180.0, "speed": 0.3})
            with svc.cond:
                svc.lose("A", why)
            o = svc.outcome()
            assert svc.reason == why and o.winner is None and svc.m.phase().state != FINISHED
            ff = svc.m.final_fields()
            assert ff["clock_out"]["A"] is (why == "clock") and not ff["complete"]["A"]
            if why == "clock":
                assert ff["scores"]["A"]["holes"] == [7, 7, 7, 7] and svc.m.scores() == {"A": 28}  # the rest score 7
            finish(svc)


def test_exploit_no_world_state():
    c = cfg(seed=987654)
    m = SPORT.build(c)
    m.apply("A", putt(aim=-5.0, speed=2.5))
    obs = json.dumps(m.observe("A"), default=float)
    text = obs + json.dumps(SPORT.spec(c, "A", m), default=float)
    assert "987654" not in text and '"seed"' not in text
    assert "executed" not in obs and "trail" not in obs and '"start"' not in obs and '"end"' not in obs
    G = m.S.G
    for H in G.HOLES:  # no layout numbers: cup, ball, slopes, bumps
        for v in [*H["cup"], *H["ball"][1:], *H["tilt"], *[q for b in H["bumps"] for q in b[:3]]]:
            if abs(v) > 1e-9:
                assert f"{v:.4f}" not in text, v
    x, y = m.M.c["A"].pos
    assert f"{x:.4f}" not in text and f"{y:.4f}" not in text


def test_exploit_pick_up_after_six():
    m = SPORT.build(cfg())
    for n in range(1, 7):
        res = m.apply("A", DRIBBLE)["result"]
        assert res["hole"] == 1 and res["stroke"] == n and res["hole_finished"] == (n == 6)
    assert res["hole_score"] == 7 and res["picked_up"] is True
    assert m.observe("A")["decision"]["hole"] == 2 and m.scores() == {"A": 7}


def test_exploit_ball_stays_on_green():
    m = SPORT.build(cfg(seed=5))
    G = m.S.G
    k = 0
    while m.phase().state != FINISHED:
        c = m.M.c["A"]
        H = G.HOLES[c.hole]
        m.apply("A", putt(aim=(k * 37.0) % 360.0 - 180.0, speed=6.0))
        k += 1
        c = m.M.c["A"]
        if c.done:
            break
        H = G.HOLES[c.hole]
        x, y = c.pos
        assert math.isfinite(x) and math.isfinite(y)
        assert -1e-6 <= x <= H["L"] + 1e-6 and abs(y) <= H["W"] / 2 + 1e-6, (k, c.pos)
        for x0, x1, y0, y1 in H["blocks"]:
            assert not (x0 < x < x1 and y0 < y < y1), (k, c.pos)
    assert k >= 4
