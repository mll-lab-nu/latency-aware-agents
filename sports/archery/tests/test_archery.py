"""archery's own tests: one per [[exploits]] entry of sport.toml (tools/check runs them through core/testkit.py, after
the generic suite). Run alone: python3 -m core.testkit tests sports/archery/plugin.py (in the sport's image)."""
import os
import tempfile
import time
from pathlib import Path

from core.episode import load_plugin
from core.spec import load
from core.sport import FINISHED, MatchConfig, Refused

PLUGIN = str(Path(__file__).resolve().parents[1] / "plugin.py")
SPORT = load_plugin(PLUGIN)
DECL = load(PLUGIN)
import archery as A  # noqa: E402  (the plugin's directory is on sys.path now)


def cfg(seed=1):
    return MatchConfig(sport="archery", format="single", seats=("A",), seed=seed, first="A", action_level="low")


def shot(el=2.95, az=0.1, draw=1.0):
    return {"type": "shoot", "elev_deg": el, "azim_deg": az, "draw": draw}


def test_exploit_bad_actions_are_refused():
    m, layer = SPORT.build(cfg()), SPORT.action_layer("low")
    d0 = m.state_digest()
    for body in (shot(el=31), shot(az=-20.5), shot(draw=0.4), shot(draw=1.01), {"type": "shoot", "elev_deg": 3.0},
                 shot(el=float("nan")), shot(el="3"), shot(draw=True), {"type": "throw", "yaw": 0},
                 {**shot(), "spin": 1}, [1, 2, 3]):
        try:
            layer.parse("A", body, m, {})
            raise AssertionError(f"accepted {body}")
        except Refused as e:
            assert e.code == 400
    assert m.state_digest() == d0 and m.k == 0
    assert layer.parse("A", shot(el=2.123456789), m, {})["elev_deg"] == 2.1235     # 4 decimals, as the legacy service


def test_exploit_out_of_turn():
    from core.turn import TurnService
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    with tempfile.TemporaryDirectory() as tmp:
        svc = TurnService(SPORT, PLUGIN, cfg(), Path(tmp))
        post = svc.routes(True)
        assert post["act"]("A", shot())[0] == 409                    # before the start: refused
        post["start"]("A", {})
        code, out = post["act"]("A", shot())
        assert code == 200 and svc.m.k == 1 and out["last_arrow"]["score"]
        assert post["act"]("A", shot())[0] == 409                    # the review comes first
        assert post["ready"]("A", {"why": "w", "plan": "p"})[0] == 200
        assert post["act"]("A", shot())[0] == 200 and svc.m.k == 2
        svc.collect()


def test_exploit_no_wind_or_positions_in_observation():
    for c in (cfg(seed=20260929), cfg(seed=2)):
        m = SPORT.build(c)
        for _ in range(7):
            m.apply(m.to_move(), shot())
        text = repr(m.observe("A"))
        winds = [w for row in A.make_winds(c.seed)[0] for pair in row for w in pair]
        hits = [h for r in m.recs for h in r["hit"][1:]] + [r["r_m"] for r in m.recs if r["r_m"] is not None]
        for x in winds + hits:
            if abs(x - round(x, 1)) < 1e-9:                              # 0.0 (ground), 50.0 (the face plane) ...
                continue
            assert repr(x) not in text and f"{x:.3f}" not in text, x
        for word in ("wind", "hit", "executed"):
            assert word not in text, word


def test_exploit_reply_image_shows_only_this_end():
    m = SPORT.build(cfg(seed=2))
    for i in range(3):
        info = m.apply("A", shot(2.9 + 0.02 * i, 0.05 * i))
        d0 = m.state_digest()
        v = m.reply_views("A", None, info)
        assert m.state_digest() == d0
        (fr,) = v["frames"]
        assert [c["name"] for c in fr["cameras"]] == ["target"]
        arrows = fr["state"]["mocap_pos"][4:7]
        shown = [p for p in arrows if p[2] != -50.0]
        assert len(shown) == i + 1, (i, arrows)                        # the third arrow too
        assert shown[-1] == [float(x) for x in m.recs[-1]["hit"]]
        socks = fr["state"]["mocap_quat"][:4]
        assert socks == plugin_lane(50.0, m.recs[-1]["wind"])["mocap_quat"][:4]   # the wind it flew in
    assert m.r.stuck == [] and m.r.end == 1                            # the end is over: arrows pulled for observe
    assert all(p[2] == -50.0 for p in m.render_state()["mocap_pos"][4:7])


def plugin_lane(D, wind):
    import plugin
    return plugin.lane_state(D, wind, [])


def test_exploit_stalling_loses():
    from core.turn import TurnService
    os.environ["RCB_PLAYER_MODE"] = "oracle"
    old = os.environ.get("CLOCK_S")
    os.environ["CLOCK_S"] = "0.2"
    try:
        for c, want in ((cfg(), None),):
            with tempfile.TemporaryDirectory() as tmp:
                svc = TurnService(SPORT, PLUGIN, c, Path(tmp))
                post = svc.routes(True)
                for s in c.seats:
                    post["start"](s, {})
                time.sleep(0.5)
                code, out = post["act"]("A", shot())
                assert code == 200 and "clock_forfeit" in out and svc.m.k == 0
                o = svc.outcome()
                assert svc.reason == "clock" and o.winner == want, (svc.reason, o)
                t = time.time()
                while not svc.finalized and time.time() - t < 120:
                    time.sleep(0.2)
                assert svc.finalized
    finally:
        if old is None:
            os.environ.pop("CLOCK_S", None)
        else:
            os.environ["CLOCK_S"] = old


def test_exploit_spec_matches_the_simulator():
    """Not a loophole but the same kind of guard: the prompt's rule constants ([spec]) are the simulator's own."""
    sp = DECL["spec"]
    assert sp["distances_m"] == A.DISTANCES and sp["arrows_per_end"] == A.ARROWS_PER_END
    assert tuple(sp["launch_point_m"]) == A.LAUNCH and sp["face_centre_z_m"] == A.FACE_Z
    assert sp["full_draw_speed_m_s"] == A.V_FULL and sp["windsock_full_speed_m_s"] == A.SOCK_FULL
    assert sp["drag_K_per_m"] == round(A.K_DRAG, 7)
    assert sp["execution_scatter_1sigma"] == {"elev_deg": A.SIG_ELEV_DEG, "azim_deg": A.SIG_AZIM_DEG,
                                              "speed_rel": A.SIG_DRAW}
    assert DECL["formats"]["single"]["success"]["score_at_least"] == 110
