"""darts' own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py, after
the generic suite). Run alone: python3 -m core.testkit tests sports/darts/plugin.py (in the sport's image)."""
import os
import tempfile
from pathlib import Path

from core import spec as spec_mod
from core.episode import load_plugin
from core.outcome import decide
from core.sport import FINISHED, MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
DECL = spec_mod.load(PLUGIN)
import board as BOARD  # noqa: E402  (the plugin's directory is on sys.path now)
import darts as D  # noqa: E402

DUEL = MatchConfig(sport="darts", format="duel", seats=("A", "B"), seed=1, first="A", action_level="low")
SINGLE = MatchConfig(sport="darts", format="501", seats=("A",), seed=1, first="A", level=2, action_level="low")
T20 = {"type": "throw", "yaw": 0.0, "pitch": 16.9, "speed": 7.0}


def test_exploit_bad_throws_are_refused():
    layer = SPORT.action_layer("low")
    for cfg in (DUEL, SINGLE):
        m = SPORT.build(cfg)
        d0 = m.state_digest()
        for body in ({**T20, "yaw": 25.0}, {**T20, "pitch": -11.0}, {**T20, "speed": 7.1}, {**T20, "speed": 4.9},
                     {"type": "throw", "yaw": 0.0, "pitch": 16.9}, {**T20, "speed": float("nan")},
                     {**T20, "yaw": "1"}, {**T20, "spin": 3.0}, {**T20, "type": "aim"}, {"yaw": 0.0}, [1, 2]):
            try:
                layer.parse("A", body, m, {"clock_left_s": {}})
                raise AssertionError(f"accepted {body}")
            except Refused as e:
                assert e.code == 400
        assert m.state_digest() == d0 and m.k == 0
        cmd = layer.parse("A", {**T20, "yaw": 0.123456789}, m, {"clock_left_s": {}})
        assert cmd["yaw"] == 0.1235                                   # 4 decimals, as the legacy service


def test_exploit_out_of_turn():
    from core.turn import TurnService
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    os.environ["RCB_BROADCAST"] = "0"
    with tempfile.TemporaryDirectory() as tmp:
        svc = TurnService(SPORT, PLUGIN, DUEL, Path(tmp))
        post = svc.routes(True)
        assert post["act"]("A", T20)[0] == 409                        # before the start: refused
        post["start"]("A", {})
        post["start"]("B", {})
        assert post["act"]("B", T20)[0] == 409                        # A throws first
        code, out = post["act"]("A", T20)
        assert code == 200 and svc.m.k == 1 and out["caller"]
        assert post["act"]("A", T20)[0] == 409                        # the review comes first
        assert post["ready"]("A", {"why": "w", "plan": "p"})[0] == 200
        for _ in range(2):                                            # A's visit: 3 darts
            assert post["act"]("A", T20)[0] == 200
            assert post["ready"]("A", {"why": "w", "plan": "p"})[0] == 200
        assert post["act"]("A", T20)[0] == 409                        # now it is B's visit
        assert svc.m.k == 3 and svc.m.to_move() == "B"
        svc.finalizing = True                                         # stop the watchdog thread's work


def test_exploit_no_hidden_state_in_observation():
    for cfg in (DUEL, SINGLE):
        m = SPORT.build(cfg)
        for _ in range(7):
            s = m.to_move()
            m.apply(s, {**T20, "clock_left_s": {"A": 3600.0, "B": 3600.0}})
        recs = m.d.log
        hidden = [m.d.k] + [x for r in recs for x in (r["u"], r["v"], *r["executed"], *r["vel"],
                                                                 r["t_flight"])]
        for seat in cfg.seats:
            obs = m.observe(seat)
            assert set(obs) <= set(DECL["observation"]["fields"]), sorted(obs)
            text = repr(obs)
            for x in hidden:
                assert repr(x) not in text and (not isinstance(x, float) or f"{x:.4f}" not in text), x
            assert "seed" not in text and "drag" not in text


def test_exploit_reply_image_is_own_visit():
    for cfg in (DUEL, SINGLE):
        m = SPORT.build(cfg)
        for n in range(4):
            seat = m.to_move()
            info = m.apply(seat, dict(T20, yaw=0.1 * n))
            d0 = m.state_digest()
            (fr,) = m.reply_views(seat, None, info)["frames"]
            assert m.state_digest() == d0 and [c["name"] for c in fr["cameras"]] == ["board"]
            mine = [r for r in m.d.log if m.side_of_rec(r) == cfg.side_of(seat) and r["leg"] == info["leg"]
                    and r["visit"] == info["visit"]]
            assert fr["state"] == BOARD.mocap(mine)
            assert len(mine) == info["dart"] and mine[-1]["dart"] == info["dart"]
            if cfg is SINGLE and n == 2 and not m.d.done:              # the opponent's visit followed inside apply
                assert m.side_of_rec(m.d.log[-1]) == "opponent" and len(mine) == 3


def test_exploit_stalling_loses():
    for why in ("clock", "abandoned", "ready_timeout"):
        m = SPORT.build(DUEL)
        m.apply("A", T20)
        m.forfeit("B", why)
        o = decide(m, "B", why, DUEL.side_list(), DUEL.side_of)
        assert m.phase().state == FINISHED and o.winner == "A", (why, o)
        m = SPORT.build(SINGLE)
        m.apply("A", T20)
        m.forfeit("A", why)
        o = decide(m, "A", why, SINGLE.side_list(), SINGLE.side_of)
        assert m.phase().state == FINISHED and o.winner is None and m.d.winner == 1, (why, o)  # 501: a loss
    m = SPORT.build(SINGLE)                                           # never throwing: the match never ends
    assert decide(m, None, None, SINGLE.side_list(), SINGLE.side_of).winner is None


def test_exploit_bust_gains_nothing():
    m = SPORT.build(DUEL)
    m.d.rem["A"] = m.d.visit["start"] = 2                             # A needs double 1; a treble-20 dart busts
    m.apply("A", T20)
    r = m.d.log[-1]
    assert r.get("event") == "bust" and m.d.rem["A"] == 2 and m.d.visits[-1]["scored"] == 0
    assert m.to_move() == "B" and m.observe("A")["remaining"]["you"] == 2


def test_exploit_seeded_world():
    ks = {D.k_drag(s) for s in (1, 2, 3, 4, 5)}
    assert len(ks) == 5 and all(D.K_MIN <= k <= D.K_MAX for k in ks)
    a, b = SPORT.build(SINGLE), SPORT.build(SINGLE)                   # the scripted opponent: the same for everyone
    assert a.state_digest() == b.state_digest() and a.d.log == b.d.log and a.d.log
    for lv in (0, 5):
        try:
            SPORT.build(MatchConfig(**{**SINGLE.to_json(), "level": lv, "seats": ("A",)}))
            raise AssertionError(f"level {lv} accepted")
        except ValueError:
            pass


def test_exploit_spec_is_the_simulator():
    sp = DECL["spec"]
    assert sp["release_point_m"] == list(D.RELEASE) and sp["gravity_m_s2"] == D.G
    assert sp["limits"] == {"yaw_deg": [-D.YAW_MAX, D.YAW_MAX], "pitch_deg": [D.PITCH_MIN, D.PITCH_MAX],
                            "speed_m_s": [D.V_MIN, D.V_MAX]}
    assert sp["execution_scatter_1sigma"] == D.PLAYER_SCATTER
    assert sp["board_face_plane_x_m"] == D.X_BOARD and sp["bull_centre_m"] == [D.X_BOARD, 0.0, D.Z_BULL]
    r = sp["radii_mm"]
    assert [r["bull"], r["outer_bull"], *r["treble_ring"], *r["double_ring"], r["board_edge"]] == [
        round(x * 1000, 2) for x in (D.R_BULL, D.R_OBULL, D.R_TRI_IN, D.R_TRI_OUT, D.R_DBL_IN, D.R_DBL_OUT, D.R_BOARD)]
    assert sp["segments_clockwise_from_top"] == D.SEGMENTS
    assert f"[{D.K_MIN}, {D.K_MAX}]" in sp["drag"] and f"{D.MAX_VISITS} visits" in sp["visit_cap"]
    assert DECL["formats"]["501"]["levels"] == sorted(D.LEVELS) and D.LEGS_TO_WIN == 2 and D.START == 501
    spec = SPORT.spec(SINGLE, "A", SPORT.build(SINGLE))
    assert spec["opponent"]["execution_scatter_1sigma"] == D.LEVELS[2]["scatter"]


def test_exploit_independent_scatter_streams():
    """Within a duel the n-th dart of each player in a leg draws from its own stream (the opponent's deviation says
    nothing about yours); the other first order on the same seed swaps the streams (the same luck over the pair)."""
    def deviations(first):
        m = D.Duel(5, first)
        out = {"A": [], "B": []}
        for _ in range(12):                                           # two visits each: darts 0-5 of each player
            r = m.throw(m.turn, 0.0, 16.9, 7.0)
            out[r["seat"]].append(tuple(round(e - c, 12) for e, c in zip(r["executed"], r["cmd"])))
        return out
    a_first, b_first = deviations("A"), deviations("B")
    assert len(a_first["A"]) == len(a_first["B"]) == 6
    assert all(x != y for x, y in zip(a_first["A"], a_first["B"]))  # independent within the match
    assert a_first["A"] == b_first["B"] and a_first["B"] == b_first["A"]   # the pair swaps the streams
    import numpy as np
    za = np.array([d[0] for s in range(1, 201) for d in deviations_seed(s)["A"]])
    zb = np.array([d[0] for s in range(1, 201) for d in deviations_seed(s)["B"]])
    assert abs(float(np.corrcoef(za, zb)[0, 1])) < 0.15                # no correlation between the n-th darts


def deviations_seed(seed):
    m = D.Duel(seed, "A")
    out = {"A": [], "B": []}
    for _ in range(6):
        r = m.throw(m.turn, 0.0, 16.9, 7.0)
        out[r["seat"]].append((r["executed"][0] - r["cmd"][0],))
    return out
