"""Bowling's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py after
the generic suite). Run alone: python3 -m core.testkit tests sports/bowling/plugin.py (in the sport's image)."""
import json
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
    return MatchConfig(sport="bowling", format="single", seats=("A",), seed=seed, first="A", action_level="low")


def roll(board=29.5, angle=-4.2, speed=8.5, rpm=450.0, axis=60.0, clocks=None):
    return {"type": "roll", "board": board, "angle_deg": angle, "speed": speed, "rpm": rpm, "axis_deg": axis,
            "clock_left_s": clocks or {"A": 3600.0}}


GUTTER = roll(board=2.0, angle=-3.0, speed=8.0, rpm=0.0, axis=0.0)


def service(tmp):
    from core.turn import TurnService
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    return TurnService(SPORT, PLUGIN, cfg(), Path(tmp))


def test_exploit_bad_actions_are_refused():
    m, layer = SPORT.build(cfg()), SPORT.action_layer("low")
    S = m.S
    # the declared limits are the simulator's, and so are the published scatter and the success line
    assert {k: (f["min"], f["max"]) for k, f in m.schema("A")["roll"].items()} == S.LIMITS
    sc = DECL["spec"]["execution_scatter_1sigma"]
    assert (sc["board"], sc["angle_deg"], sc["speed_rel"], sc["rpm_rel"], sc["axis_deg"]) == tuple(
        S.B.SIG[k] for k in ("board", "angle_deg", "speed_frac", "rpm_frac", "axis_deg"))
    assert DECL["formats"]["single"]["success"]["score_at_least"] == S.SUCCESS_MIN_TOTAL
    good = {"type": "roll", "board": 20, "angle_deg": 0, "speed": 8, "rpm": 300, "axis_deg": 30}
    d0 = m.state_digest()
    for body in ({**good, "board": 39.5}, {**good, "speed": 2.0}, {**good, "rpm": float("nan")},
                 {**good, "speed": 4.46}, {**good, "speed": 10.74},          # bowling-4: E.A.R.L.'s 4.47-10.73 m/s
                 {**good, "axis_deg": "45"}, {k: v for k, v in good.items() if k != "rpm"}, {**good, "type": "throw"},
                 {**good, "ball_x": 18.0}, {**good, "angle_deg": float("inf")}, [1, 2]):
        try:
            layer.parse("A", body, m, {"clock_left_s": {"A": 1.0}})
            raise AssertionError(f"accepted {body}")
        except Refused as e:
            assert e.code == 400
    assert m.state_digest() == d0 and m.k == 0
    cmd = layer.parse("A", {**good, "board": 20.123456}, m, {"clock_left_s": {"A": 12.34567}})
    assert cmd["board"] == 20.1235 and cmd["clock_left_s"] == {"A": 12.346}               # the legacy rounding
    cmd = layer.parse("A", good, m, {"clock_left_s": {"A": -1}})
    assert cmd["clock_left_s"] == {"A": 0.0}


def test_exploit_out_of_turn():
    with tempfile.TemporaryDirectory() as tmp:
        svc = service(tmp)
        post = svc.routes(True)
        body = {"type": "roll", "board": 29.5, "angle_deg": -4.2, "speed": 8.5, "rpm": 450, "axis_deg": 60}
        assert post["act"]("A", body)[0] == 409                      # before the start: refused
        post["start"]("A", {})
        code, out = post["act"]("A", body)
        assert code == 200 and svc.m.k == 1 and "pins_down" in out["result"]
        assert post["act"]("A", body)[0] == 409                      # the review comes first
        assert svc.m.k == 1
        assert post["ready"]("A", {"why": "w", "plan": "p"})[0] == 200
        assert post["act"]("A", body)[0] == 200 and svc.m.k == 2


def finish(svc):
    import time
    for _ in range(600):                                             # the fresh-process replay, before the cleanup
        if svc.finalized:
            return
        time.sleep(0.1)


def test_exploit_stalling_loses():
    for why in ("clock", "abandoned"):                               # a game ended early is a valid failure
        with tempfile.TemporaryDirectory() as tmp:
            svc = service(tmp)
            post = svc.routes(True)
            post["start"]("A", {})
            post["act"]("A", {"type": "roll", "board": 29.5, "angle_deg": -4.2, "speed": 8.5, "rpm": 450,
                              "axis_deg": 60})
            with svc.cond:
                svc.lose("A", why)
            o = svc.outcome()
            assert svc.reason == why and o.winner is None and svc.m.phase().state != FINISHED
            assert svc.m.final_fields()["clock_out"]["A"] is (why == "clock")
            assert not svc.m.final_fields()["complete"]["A"]
            finish(svc)


def test_exploit_no_world_state():
    c = cfg(seed=987654)
    m = SPORT.build(c)
    m.apply("A", roll())
    if m.to_move() == "A":
        m.apply("A", roll(board=20.0, angle=0.0, rpm=0.0, axis=0.0))
    obs = json.dumps(m.observe("A"), default=float)
    text = obs + json.dumps(SPORT.spec(c, "A", m), default=float)
    assert "987654" not in text and '"seed"' not in text
    lane = m.S.B.LANE
    assert not any(k in text for k in ("oil_len", "mu_oil", "mu_dry", "taper"))
    assert repr(lane["oil_len"]) not in text and repr(lane["mu_dry"]) not in text
    assert "executed" not in obs and "trail" not in obs                 # the ball's real release and path
    g = m.M.cur("A")
    for i in range(10):
        p, _ = g.lane.pin_pose(i)
        for v in p[:2]:
            assert f"{v:.3f}" not in obs and f"{v:.4f}" not in obs, v


def test_exploit_gutter_respots():
    m = SPORT.build(cfg())
    g = m.M.cur("A")
    q0 = [g.lane.pin_pose(i)[0].copy() for i in range(10)]
    res = m.apply("A", GUTTER)["result"]
    assert res["gutter"] and res["pins_down"] == 0 and m.observe("A")["decision"]["pins_standing"] == 10
    for i in range(10):
        assert abs(g.lane.pin_pose(i)[0] - q0[i]).max() < 1e-9          # the deck as it was before the ball
    assert m.scores() == {"A": 0}
